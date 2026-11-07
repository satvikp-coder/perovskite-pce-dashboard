"""Load model_store/ and predict single compositions. Used by the dashboard; reads only from model_store/."""
import os, json
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from . import core as C

DEFAULT_STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_store")


def load_store(path=DEFAULT_STORE):
    S = dict(path=path)
    S["meta"] = json.load(open(os.path.join(path, "meta.json")))
    S["cats"] = json.load(open(os.path.join(path, "categories.json")))
    S["stab_cats"] = json.load(open(os.path.join(path, "categories_stability.json")))
    S["models"] = {}
    for k in C.TARGETS + ["STAB"]:
        m = XGBRegressor(); m.load_model(os.path.join(path, f"model_{k}.json")); S["models"][k] = m
    for name in ["candidates", "metrics_targets", "cv_blocked_pce", "leave_family_out", "C_shap_all_features_pce", "pdp_top5", "stability_scenarios",
                 "stability_ablation", "test_predictions_random", "test_predictions_blocked", "E_database_compositions"]:
        S[name] = pd.read_csv(os.path.join(path, f"{name}.csv"))
    mc = os.path.join(path, "results", "model_comparison_pce.csv"); S["model_comparison"] = pd.read_csv(mc) if os.path.exists(mc) else None
    S["shrinkage"] = json.load(open(os.path.join(path, "pce_shrinkage.json")))
    S["stab_facts"] = json.load(open(os.path.join(path, "stability_data_facts.json")))
    dbi = pd.read_csv(os.path.join(path, "db_index.csv"))
    S["db_M"] = dbi[[f"{s}_{k}" for s, k in C.VEC_IONS]].values
    S["db_name"], S["db_pce"] = dbi["Name"].fillna("").values, dbi["PCE"].values
    by_key = {}
    for i, key in enumerate(C.ionset_key(S["db_M"])): by_key.setdefault(key, []).append(i)
    S["db_by_key"] = by_key
    z = np.load(os.path.join(path, "shap_sample.npz"), allow_pickle=False)
    S["shap"], S["shap_X"], S["shap_features"] = z["shap"], z["X"], list(z["features"])
    return S


def validate_fractions(site):
    """site: {ion: fraction}. Returns error string or None. Fractions must be in [0,1] and sum to 1 (tolerance 0.001)."""
    if any(v < 0 or v > 1 for v in site.values()): return "fractions must be between 0 and 1"
    s = sum(site.values())
    return None if abs(s - 1.0) <= 1e-3 else f"fractions sum to {s:.3f}, must be 1.000"


def _site(d):
    items = [(k, v) for k, v in d.items() if v > 1e-9]
    return tuple(k for k, _ in items), [v for _, v in items]


def make_row(A, B, X, etl, htl, contact, dep_atm, jv_atm, additives, area_cm2):
    (sa, ga), (sb, gb), (sx, gx) = _site(A), _site(B), _site(X)
    r = C.composition_row(sa, ga, sb, gb, sx, gx)
    r.update(C.CTX)
    r["Cell_stack_sequence"] = f"SLG | FTO | {etl} | Perovskite | {htl} | {contact}"
    r["Perovskite_additives_compounds"] = "; ".join(additives) if additives else "Undoped"
    r["Perovskite_deposition_synthesis_atmosphere"] = dep_atm; r["JV_test_atmosphere"] = jv_atm; r["Cell_area_measured"] = float(area_cm2)
    return pd.DataFrame([r])


def predict(S, row):
    X = C.build_features(row, cats=S["cats"])[S["meta"]["features"]]
    p = {t: float(S["models"][t].predict(X)[0]) for t in C.TARGETS}
    p["PCE_physics"] = float(C.pce_physics(p["Voc"], p["Jsc"], p["FF"]))
    p["Confidence"] = float(C.confidence(p["PCE"], p["PCE_physics"]))
    p["tolerance_factor"], p["octahedral_factor"] = float(X.tolerance_factor[0]), float(X.octahedral_factor[0])
    p["plausible"] = bool(C.TOLERANCE_WINDOW[0] <= p["tolerance_factor"] <= C.TOLERANCE_WINDOW[1] and C.OCTAHEDRAL_WINDOW[0] <= p["octahedral_factor"] <= C.OCTAHEDRAL_WINDOW[1])
    vec = C.vec_cols(X)
    st, dist, nsame = C.novelty_check(vec, S["db_M"], S["db_by_key"], C.NOVELTY_TOL)
    p["novelty"], p["nearest_distance"], p["n_same_ionset"] = st[0], dist[0], nsame[0]
    p["extrapolation"] = ("HIGH (ion-set never seen)" if st[0].startswith("new ion-set") else
                          "MEDIUM (far from seen ratios)" if dist[0] > 0.15 else "LOW")
    # stability proxy (test duration, NOT time-to-80%)
    st_row = row.assign(Stability_atmosphere="Air", Stability_light_intensity=100.0)
    Xs = C.build_features(st_row, cats={**S["cats"], **S["stab_cats"]}, stability=True)[S["meta"]["stability_features"]]
    p["stab_proxy_h"] = float(np.expm1(S["models"]["STAB"].predict(Xs)[0]))
    p["nearest"] = nearest(S, vec[0])
    return p


def nearest(S, v, k=5):
    D = np.abs(S["db_M"] - v).max(1); seen, out = set(), []
    for i in np.argsort(D):
        key = tuple(np.round(S["db_M"][i], 3))
        if key in seen: continue
        seen.add(key)
        same = np.where(np.abs(S["db_M"] - S["db_M"][i]).max(1) < 1e-6)[0]
        out.append(dict(Name=S["db_name"][i], Distance=round(float(D[i]), 3), N_devices=len(same), Mean_PCE=round(float(np.nanmean(S["db_pce"][same])), 2)))
        if len(out) >= k: break
    return out


def reliability(p):
    """Traffic light for a single prediction. Rule of thumb (not a calibrated probability): uses the direct-vs-physics agreement,
    extrapolation level and the chemistry window. Returns (level, one-sentence reason)."""
    ex = p["extrapolation"].split()[0]
    if ex == "HIGH" or not p["plausible"] or p["Confidence"] < 0.6:
        why = ("this ion combination never appears in the database" if ex == "HIGH" else
               "the composition is outside the usual structural-stability window" if not p["plausible"] else
               "the two internal prediction routes disagree")
        return "Low", f"Treat as a rough hypothesis: {why}."
    if ex == "LOW" and p["Confidence"] >= 0.85:
        return "High", "Many similar compositions exist in the database and the two internal prediction routes agree."
    return "Medium", ("The ratios are somewhat far from compositions seen in the database." if ex == "MEDIUM" else
                      "The two internal prediction routes agree only moderately.")
