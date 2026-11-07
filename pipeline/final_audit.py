"""Final correctness audit. Reads raw data (read-only) and model_store/. Writes model_store/results/final_audit.csv.
Retrains only temporary split models (not saved)."""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from pv2 import core as C, store as ST
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); MS = f"{ROOT}/model_store"
RAW = os.path.join(os.path.dirname(ROOT), "merged_perovskite.csv")
rows = []
def add(check, item, value, status, note=""):
    rows.append(dict(check=check, item=item, value=value, status=status, note=note)); print(check, item, value, status, note, flush=True)

meta = json.load(open(f"{MS}/meta.json")); F = meta["features"]
# 1 leakage
try: C.assert_no_leakage(F); add("1 leakage", "PCE feature list free of Voc/Jsc/FF/PCE columns", f"{len(F)} features", "PASS")
except AssertionError as e: add("1 leakage", "feature list", str(e), "FAIL")
raw = C.read_csv_safe(RAW); X, cats = C.build_features(raw); assert list(X.columns) == F
idx = C.valid_rows(raw); tr_r, te_r = C.random_split(idx); G = C.group_key(raw, C.BLOCK_RAW); tr_g, te_g = C.blocked_split(idx, G)
add("1 leakage", "build_features columns == saved feature list", len(F), "PASS")
key = pd.util.hash_pandas_object(X.round(6), index=False).values
for nm, tr, te in (("random", tr_r, te_r), ("blocked", tr_g, te_g)):
    pct = float(np.isin(key[te], key[tr]).mean() * 100)
    add("1 leakage", f"{nm}: % test rows whose full feature vector also in train", round(pct, 2), "INFO",
        "same feature vector = same composition+stack+conditions (different devices); inflates random-split score")
d = pd.Series(key[idx]).duplicated().mean() * 100
add("1 leakage", "% valid rows with a duplicated feature vector (any)", round(float(d), 2), "INFO")
add("1 leakage", "exact duplicate raw rows (valid rows)", int(raw.iloc[idx].duplicated().sum()), "INFO")
ov = len(set(G[tr_g]) & set(G[te_g]))
add("1 leakage", "blocked split groups disjoint (overlap count)", ov, "PASS" if ov == 0 else "FAIL")
# 2 metrics recompute (temporary models trained on the split's train part)
ref = pd.read_csv(f"{MS}/metrics_targets.csv")
Y = {"PCE": raw.JV_default_PCE.values, "Voc": raw.JV_default_Voc.values, "Jsc": raw.JV_default_Jsc.values, "FF": raw.JV_default_FF.values}
best = meta["best_config"]
for sp, (tr, te) in {"random 80/20": (tr_r, te_r), "blocked 80/20": (tr_g, te_g)}.items():
    P = {t: C.make_xgb(**best).fit(X.iloc[tr], Y[t][tr]).predict(X.iloc[te]) for t in C.TARGETS}
    P["phys"] = C.pce_physics(P["Voc"], P["Jsc"], P["FF"])
    tp = pd.read_csv(f"{MS}/test_predictions_{sp.split()[0]}.csv")
    for t in C.TARGETS + ["PCE from Voc x Jsc x FF"]:
        ph = t.startswith("PCE from"); pr = P["phys"] if ph else P[t]; yt = Y["PCE" if ph else t][te]
        m = C.metrics(yt, pr, min_y=1.0 if t.startswith("PCE") else None); r = ref[(ref.Target == t) & (ref.Split == sp)].iloc[0]
        dR = abs(m["R2"] - r.R2); dM = abs(m["MAE"] - r.MAE)
        add("2 metrics", f"{t} {sp} recomputed R2/MAE vs file", f"{m['R2']:.4f}/{m['MAE']:.4f} vs {r.R2:.4f}/{r.MAE:.4f}",
            "PASS" if max(dR, dM) <= 0.005 else "FAIL", f"dR2={dR:.4f} dMAE={dM:.4f}")
        if t in C.TARGETS:
            m2 = C.metrics(tp[f"actual_{t}"], tp[f"pred_{t}"], min_y=1.0 if t == "PCE" else None)
            ok = abs(m2["R2"] - r.R2) <= 1e-6 and abs(m2["MAE"] - r.MAE) <= 1e-6
            add("2 metrics", f"{t} {sp} test_predictions csv reproduces metrics csv (dashboard source)", f"{m2['R2']:.4f}", "PASS" if ok else "FAIL")
# 3 sanity predictions (saved final models)
S = ST.load_store()
def pred(A, B, Xs): return ST.predict(S, ST.make_row(A, B, Xs, "TiO2-c", "Spiro-MeOTAD", "Au", "N2", "Air", [], 0.1))
cases = {"MAPbI3": ({"MA": 1.0}, {"Pb": 1.0}, {"I": 1.0}), "FAPbI3": ({"FA": 1.0}, {"Pb": 1.0}, {"I": 1.0}), "CsPbBr3": ({"Cs": 1.0}, {"Pb": 1.0}, {"Br": 1.0}),
         "CsPbI3": ({"Cs": 1.0}, {"Pb": 1.0}, {"I": 1.0}), "MASnI3": ({"MA": 1.0}, {"Sn": 1.0}, {"I": 1.0}), "FASnI3": ({"FA": 1.0}, {"Sn": 1.0}, {"I": 1.0})}
R = {k: pred(*v) for k, v in cases.items()}
for k, p in R.items():
    exact = p["PCE_physics"] == p["Voc"] * p["Jsc"] * p["FF"]
    add("3 sanity", f"{k} PCE direct/physics/Voc/Jsc/FF", f"{p['PCE']:.2f}/{p['PCE_physics']:.2f}/{p['Voc']:.3f}/{p['Jsc']:.2f}/{p['FF']:.3f}", "INFO", f"extrap={p['extrapolation']}")
    add("3 sanity", f"{k} physics PCE == Voc*Jsc*FF exactly", exact, "PASS" if exact else "FAIL")
vm = np.isin(np.arange(len(raw)), idx); mapi = raw.Perovskite_composition_short_form.eq("MAPbI").values & vm
mm = float(np.nanmean(Y["PCE"][mapi]))
add("3 sanity", "database mean PCE (valid rows) / MAPbI mean", f"{np.nanmean(Y['PCE'][idx]):.2f} / {mm:.2f}", "INFO")
add("3 sanity", "MAPbI3 PCE within 3 pp of database MAPbI mean", f"{R['MAPbI3']['PCE']:.2f}", "PASS" if abs(R["MAPbI3"]["PCE"] - mm) < 3 else "FLAG")
add("3 sanity", "FAPbI3 > CsPbBr3", f"{R['FAPbI3']['PCE']:.2f} > {R['CsPbBr3']['PCE']:.2f}", "PASS" if R["FAPbI3"]["PCE"] > R["CsPbBr3"]["PCE"] else "FLAG")
for a_, b_ in (("MASnI3", "MAPbI3"), ("FASnI3", "FAPbI3")):
    add("3 sanity", f"Sn case < Pb case: {a_} < {b_}", f"{R[a_]['PCE']:.2f} < {R[b_]['PCE']:.2f}", "PASS" if R[a_]["PCE"] < R[b_]["PCE"] else "FLAG")
# 4 candidates
cand = pd.read_csv(f"{MS}/candidates.csv"); dbi = pd.read_csv(f"{MS}/db_index.csv"); Mdb = dbi[[f"{s}_{k}" for s, k in C.VEC_IONS]].values
by = {}
for i, k in enumerate(C.ionset_key(Mdb)): by.setdefault(k, []).append(i)
AB = cand[cand.List.isin(["A", "B"])]
Xc = C.build_features(AB.assign(**C.CTX, Perovskite_composition_inorganic="False", Perovskite_composition_leadfree="False"), cats=cats)
st_, dist, _ = C.novelty_check(C.vec_cols(Xc), Mdb, by, 0.03)
bad = sum(s.startswith("already") for s in st_); badcol = int(AB.Novelty_status.str.startswith("already").sum())
add("4 candidates", "List A/B rows recomputed 'already in database' (canonical order, tol 0.03)", f"{bad} of {len(AB)}", "PASS" if bad == 0 and badcol == 0 else "FAIL",
    f"stored 'already' in A/B: {badcol}; lists={AB.List.value_counts().to_dict()}")
hits = []
for dp, _, fs in os.walk(ROOT):
    if any(x in dp for x in ("results_v2_ref", "runs", "artifacts", "__pycache__", "data_derived", ".git", "tests")): continue
    for f in fs:
        if f.endswith((".py", ".md", ".csv", ".txt", ".toml", ".yaml", ".yml")) and not f.startswith(("0", "1")) and f not in ("final_audit.py", "final_audit.csv", "PROGRESS.md"):
            t = open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read()
            for m in re.finditer(r"[^\n]*\bnovel\b[^\n]*", t, re.I):
                if not re.search(r"not a novelty claim|never .{0,3}novel|does not mean novel|no novelty|not .{0,25}novel|novelty = 'not in database'", m.group(0), re.I): hits.append(f"{f}: {m.group(0)[:90]}")
hits = [h for h in hits if not h.startswith("train_export") and not h.startswith("core.py")]   # code identifiers (Novelty_status / novelty_check)
add("4 candidates", "'novel' wording outside negations (app, CSVs, README)", len(hits), "PASS" if not hits else "FLAG", " || ".join(hits[:6]))
# 5 tolerance factor
hand = {"MAPbI3": (2.17, 1.19, 2.20), "FAPbI3": (2.53, 1.19, 2.20), "CsPbI3": (1.88, 1.19, 2.20), "CsPbBr3": (1.88, 1.19, 1.96), "CsSnI3": (1.88, 1.10, 2.20)}
AS = {"MAPbI3": {"MA": 1.0}, "FAPbI3": {"FA": 1.0}, "CsPbI3": {"Cs": 1.0}, "CsPbBr3": {"Cs": 1.0}, "CsSnI3": {"Cs": 1.0}}
for k, (a_, b_, x_) in hand.items():
    h = (a_ + x_) / (2 ** 0.5 * (b_ + x_))
    row = ST.make_row(AS[k], {"Sn": 1.0} if "Sn" in k else {"Pb": 1.0}, {"Br": 1.0} if "Br" in k else {"I": 1.0}, "TiO2-c", "Spiro-MeOTAD", "Au", "N2", "Air", [], 0.1)
    code = float(C.build_features(row, cats=cats).tolerance_factor[0])
    add("5 tolerance", f"{k} hand vs code", f"{h:.4f} vs {code:.4f}", "PASS" if abs(h - code) < 1e-9 else "FAIL")
pd.DataFrame(rows).to_csv(f"{MS}/results/final_audit.csv", index=False)
print("FAIL/FLAG:", [r["item"] for r in rows if r["status"] in ("FAIL", "FLAG")])
