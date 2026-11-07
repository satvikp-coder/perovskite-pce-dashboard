"""Train tuned XGBoost models (PCE, Voc, Jsc, FF, stability proxy) and export everything the dashboard needs to model_store/.
Usage: python pipeline/train_export.py --raw ../merged_perovskite.csv --p3 data_derived/paper3_stability_rebuilt.csv [--out model_store] [--smoke]
Raw data are only read. PCE models never receive Voc/Jsc/FF."""
import os, sys, json, time, argparse, itertools, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
from sklearn.inspection import partial_dependence
import shap
from pv2 import core as C

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True); ap.add_argument("--p3", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_store"))
ap.add_argument("--smoke", type=int, default=0); ap.add_argument("--stab-only", action="store_true", help="retrain only the stability model + stability-dependent outputs; PCE/Voc/Jsc/FF files untouched")
a = ap.parse_args(); OUT, SMOKE = a.out, a.smoke
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(*x): print(f"[{time.time()-T0:6.0f}s]", *x, flush=True)
def sj(obj, name):
    with open(os.path.join(OUT, name), "w") as f: json.dump(obj, f, indent=1, default=float)
mk = lambda **kw: C.make_xgb(smoke=bool(SMOKE), **kw)

raw = C.read_csv_safe(a.raw); p3 = C.read_csv_safe(a.p3)
if SMOKE: raw = raw.sample(6000, random_state=C.SEED).reset_index(drop=True); p3 = p3.sample(2500, random_state=C.SEED).reset_index(drop=True)
raw = raw.reset_index(drop=True); p3 = p3.reset_index(drop=True)
X_all, CATS = C.build_features(raw); FEATURES = list(X_all.columns); C.assert_no_leakage(FEATURES)
GROUPS = C.group_key(raw, C.BLOCK_RAW); idx_ok = C.valid_rows(raw)
Y = {"PCE": raw.JV_default_PCE.values, "Voc": raw.JV_default_Voc.values, "Jsc": raw.JV_default_Jsc.values, "FF": raw.JV_default_FF.values}
tr_r, te_r = C.random_split(idx_ok); tr_g, te_g = C.blocked_split(idx_ok, GROUPS)
assert len(set(GROUPS[tr_g]) & set(GROUPS[te_g])) == 0
log(f"{len(FEATURES)} features, {len(idx_ok)} valid rows, random {len(tr_r)}/{len(te_r)}, blocked {len(tr_g)}/{len(te_g)}")

STABONLY = bool(a.stab_only)
if STABONLY:
    META = json.load(open(f"{OUT}/meta.json")); BEST = META["best_config"]
    FINAL = {t: mk(**BEST) for t in C.TARGETS}
    for t_, m_ in FINAL.items(): m_.load_model(f"{OUT}/model_{t_}.json")
    log("stab-only: loaded existing PCE/Voc/Jsc/FF models")
else:
    # --- tuning on blocked validation (PCE) ---
    CONFIGS = C.CONFIGS[:2] if SMOKE else C.CONFIGS
    a_, b_ = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=C.SEED).split(tr_g, groups=GROUPS[tr_g]))
    tune = []
    for i, cfg in enumerate(CONFIGS):
        m = mk(**cfg).fit(X_all.iloc[tr_g[a_]], Y["PCE"][tr_g[a_]])
        tune.append(dict(config=i + 1, **cfg, valid_R2_blocked=round(r2_score(Y["PCE"][tr_g[b_]], m.predict(X_all.iloc[tr_g[b_]])), 4)))
    tune = pd.DataFrame(tune); tune.to_csv(f"{OUT}/tuning_search.csv", index=False)
    BEST = CONFIGS[int(tune.valid_R2_blocked.values.argmax())]; log("best config", BEST)

    # --- metrics for 4 targets + physics PCE, random and blocked ---
    res, preds = [], {}
    for sp, (tr, te) in {"random 80/20": (tr_r, te_r), "blocked 80/20": (tr_g, te_g)}.items():
        P = {}
        for t in C.TARGETS:
            P[t] = mk(**BEST).fit(X_all.iloc[tr], Y[t][tr]).predict(X_all.iloc[te])
            res.append(dict(Target=t, Split=sp, **C.metrics(Y[t][te], P[t], min_y=(1.0 if t == "PCE" else None))))
        P["PCE_physics"] = C.pce_physics(P["Voc"], P["Jsc"], P["FF"])
        res.append(dict(Target="PCE from Voc x Jsc x FF", Split=sp, **C.metrics(Y["PCE"][te], P["PCE_physics"], min_y=1.0)))
        P["PCE_mean"] = 0.5 * (P["PCE"] + P["PCE_physics"])
        res.append(dict(Target="PCE average (direct + physics)", Split=sp, **C.metrics(Y["PCE"][te], P["PCE_mean"], min_y=1.0)))
        preds[sp] = (te, P)
    resdf = pd.DataFrame(res); resdf.to_csv(f"{OUT}/metrics_targets.csv", index=False); log("\n" + resdf.round(3).to_string())
    for sp, (te_, P_) in preds.items():
        d_ = pd.DataFrame({f"actual_{t}": Y[t][te_] for t in C.TARGETS})
        for k_, v_ in P_.items(): d_[f"pred_{k_}"] = v_
        d_.to_csv(f"{OUT}/test_predictions_{'random' if sp.startswith('random') else 'blocked'}.csv", index=False)
    te_s, P_s = preds["blocked 80/20"]; y_s = Y["PCE"][te_s]
    sj(dict(slope_pred_vs_actual_blocked=float(np.polyfit(y_s, P_s["PCE"], 1)[0]), max_pred_testset=float(P_s["PCE"].max()), max_actual_testset=float(y_s.max())), "pce_shrinkage.json")

    # --- blocked 5-fold CV ---
    cv = []
    for k, (a1, b1) in enumerate(GroupKFold(n_splits=3 if SMOKE else 5).split(idx_ok, groups=GROUPS[idx_ok])):
        m = mk(**BEST).fit(X_all.iloc[idx_ok[a1]], Y["PCE"][idx_ok[a1]]); cv.append(r2_score(Y["PCE"][idx_ok[b1]], m.predict(X_all.iloc[idx_ok[b1]])))
    pd.DataFrame(dict(Fold=range(1, len(cv) + 1), R2=cv)).to_csv(f"{OUT}/cv_blocked_pce.csv", index=False); log(f"blocked CV {np.mean(cv):.3f} +- {np.std(cv):.3f}")

    # --- leave-one-family-out ---
    sf = raw.Perovskite_composition_short_form.fillna("")
    FAM = {"FAPbI": ["FAPbI"], "CsPbBrI": ["CsPbBrI"], "CsFAPbBrI": ["CsFAPbBrI"], "FAMAPbI": ["FAMAPbI"],
           "Rb-quaternary (CsFAMARbPbI + CsFAMARbPbBrI)": ["CsFAMARbPbI", "CsFAMARbPbBrI"], "Sn-based (MAPbSnI etc.)": [s for s in sf.unique() if "Sn" in s and "Pb" in s][:5]}
    lofo = []
    for name, members in FAM.items():
        te = np.array([i for i in idx_ok if sf.values[i] in members]); tr = np.array([i for i in idx_ok if sf.values[i] not in members])
        if len(te) < 5: continue
        p = mk(**BEST).fit(X_all.iloc[tr], Y["PCE"][tr]).predict(X_all.iloc[te])
        lofo.append(dict(Family=name, N_devices=len(te), Actual_mean_PCE=Y["PCE"][te].mean(), Predicted_mean_PCE=p.mean(), Bias=p.mean() - Y["PCE"][te].mean(),
                         MAE=mean_absolute_error(Y["PCE"][te], p), R2=(r2_score(Y["PCE"][te], p) if len(te) >= 30 else np.nan)))
    pd.DataFrame(lofo).to_csv(f"{OUT}/leave_family_out.csv", index=False)

    # --- final models (all valid rows) ---
    FINAL = {t: mk(**BEST).fit(X_all.iloc[idx_ok], Y[t][idx_ok]) for t in C.TARGETS}; log("final models trained")

    # --- SHAP + partial dependence (model on random-split train) ---
    mS = mk(**BEST).fit(X_all.iloc[tr_r], Y["PCE"][tr_r])
    samp = X_all.iloc[te_r].sample(min(2000, len(te_r)), random_state=C.SEED)
    sv = shap.TreeExplainer(mS).shap_values(samp)
    tab = pd.DataFrame(dict(Feature=FEATURES, Mean_abs_SHAP=np.abs(sv).mean(0)))
    def direction(f, j):
        if not np.nanstd(samp[f]) > 0: return "no clear direction"
        r = np.corrcoef(np.nan_to_num(samp[f].values, nan=np.nanmedian(samp[f])), sv[:, j])[0, 1]
        return "higher value -> higher PCE" if r > 0.05 else "higher value -> lower PCE" if r < -0.05 else "no clear direction"
    tab["Direction"] = [direction(f, j) for j, f in enumerate(FEATURES)]
    tab = tab.sort_values("Mean_abs_SHAP", ascending=False).reset_index(drop=True); tab.insert(0, "Rank", tab.index + 1)
    tab.to_csv(f"{OUT}/C_shap_all_features_pce.csv", index=False)
    np.savez_compressed(f"{OUT}/shap_sample.npz", shap=sv.astype(np.float32), X=samp.values.astype(np.float32), features=np.array(FEATURES))
    top5 = [f for f in tab.Feature if f not in ("xA_other", "xB_other", "xX_other")][:5]
    pdp_rows = []
    for f in top5:
        pdp = partial_dependence(mS, samp, [f], grid_resolution=20)
        pdp_rows += [dict(Feature=f, value=float(g), predicted_PCE=float(v)) for g, v in zip(pdp["grid_values"][0], pdp["average"][0])]
    pd.DataFrame(pdp_rows).to_csv(f"{OUT}/pdp_top5.csv", index=False); log("SHAP/PDP done", top5)

# --- stability: scenarios (constant multipliers), ablation, proxy model ---
raw_st = raw[raw.Stability_time_total_exposure.notna()]
base = p3.raw_TS80.values * p3.A_light.values; rh_row = p3.Relative_Humidity.values
scen_def = [("S0  27 C, no humidity factor (v1 report)", 27, None), ("S1  50 C, RH 50%  (suggested scenario)", 50, 50), ("S2 100 C, RH 50%", 100, 50),
            ("S3 150 C, RH 50%", 150, 50), ("S4 200 C, RH 50%", 200, 50), ("S5  50 C, RH 100%", 50, 100)]
Xs_all, STAB_CATS = C.build_features(p3, stability=True); gs = C.group_key(p3, C.BLOCK_STAB)
trs, tes = train_test_split(np.arange(len(p3)), test_size=0.2, random_state=C.SEED)
srows = []
for name, tC, rh in scen_def:
    aT = float(C.arrhenius(tC + 273.15)); ah = float(C.rh_factor(rh)) if rh is not None else 1.0
    arh = np.where(~np.isnan(rh_row), C.rh_factor(np.nan_to_num(rh_row)), ah)
    ts = base * aT * arh; y = np.log1p(ts)
    r = C.metrics(y[tes], mk(**BEST).fit(Xs_all.iloc[trs], y[trs]).predict(Xs_all.iloc[tes]))
    srows.append(dict(Scenario=name, A_temperature=aT, A_humidity_default=ah, Combined_factor=aT * ah, Mean_TS80m_h=ts.mean(), Median_TS80m_h=np.median(ts),
                      Max_TS80m_h=ts.max(), Model_R2_log=r["R2"], Model_MAE_log=r["MAE"]))
pd.DataFrame(srows).to_csv(f"{OUT}/stability_scenarios.csv", index=False)
y0 = np.log1p(base); proto = [c for c in Xs_all.columns if c.startswith("stabatm_") or c.startswith("stab_")]
chem = [c for c in Xs_all.columns if c not in proto]; abl = []
for name, cols in [("Test protocol only (atmosphere + light)", proto), ("Composition + device + process only", chem), ("Both (full stability model)", list(Xs_all.columns))]:
    for sp, (a2, b2) in {"random": (trs, tes), "blocked": next(GroupShuffleSplit(1, test_size=0.2, random_state=C.SEED).split(np.arange(len(p3)), groups=gs))}.items():
        r = C.metrics(y0[b2], mk(**BEST).fit(Xs_all.iloc[a2][cols], y0[a2]).predict(Xs_all.iloc[b2][cols]))
        abl.append(dict(Inputs=name, Split=sp, R2_log=r["R2"], MAE_log=r["MAE"]))
pd.DataFrame(abl).to_csv(f"{OUT}/stability_ablation.csv", index=False)
FINAL["STAB"] = mk(**BEST).fit(Xs_all, y0)
sj(dict(n_stability_rows=int(len(raw_st)), n_numeric_temperature_values=0,
        temperature_column_values=raw_st.Stability_temperature_load_condition.fillna("(missing)").str.lower().value_counts().to_dict(),
        n_rows_with_any_humidity=int(p3.Relative_Humidity.notna().sum()), median_test_duration_h=float(raw_st.Stability_time_total_exposure.median())), "stability_data_facts.json")
log("stability done")

# --- candidates (ion-order safe, ratio aware, whole-database novelty) ---
cand = C.build_candidate_table(C.INCLUDE_GE, C.SEED, bool(SMOKE)); log(f"raw candidate grid {len(cand):,}")
Xc = C.build_features(cand, cats=CATS); ok = C.chemistry_filter(Xc)
cand, Xc = cand[ok].reset_index(drop=True), Xc[ok].reset_index(drop=True)
Mdb, by_key = C.build_db_index(X_all)
st_, dist, nsame = C.novelty_check(C.vec_cols(Xc), Mdb, by_key, C.NOVELTY_TOL)
cand["Novelty_status"], cand["Nearest_DB_ratio_distance"], cand["N_DB_devices_same_ionset"] = st_, dist, nsame
Xc_st = C.build_features(cand.assign(Stability_atmosphere="Air", Stability_light_intensity=100.0), cats={**CATS, **STAB_CATS}, stability=True)
cand["Pred_PCE_direct"] = FINAL["PCE"].predict(Xc)
cand["Pred_Voc_V"], cand["Pred_Jsc_mAcm2"], cand["Pred_FF"] = FINAL["Voc"].predict(Xc), FINAL["Jsc"].predict(Xc), FINAL["FF"].predict(Xc)
cand["Pred_PCE_physics"] = C.pce_physics(cand.Pred_Voc_V, cand.Pred_Jsc_mAcm2, cand.Pred_FF)
cand["Pred_PCE"] = (0.5 * (cand.Pred_PCE_direct + cand.Pred_PCE_physics)).clip(0, 33)
cand["Pred_stability_proxy_h"] = np.expm1(FINAL["STAB"].predict(Xc_st[Xs_all.columns])).clip(0, 50000)
cand["Confidence"] = C.confidence(cand.Pred_PCE_direct, cand.Pred_PCE_physics)
cand["Extrapolation_flag"] = np.where(cand.Novelty_status.str.startswith("new ion-set"), "HIGH (ion-set never seen)",
                              np.where(cand.Nearest_DB_ratio_distance.fillna(0) > 0.15, "MEDIUM (far from seen ratios)", "LOW"))
cand["Pareto_optimal"] = C.pareto_mask(cand.Pred_PCE.values, cand.Pred_stability_proxy_h.values)
cand["Score"] = C.composite_score(cand.Pred_PCE, cand.Pred_stability_proxy_h, cand.Confidence)
cand["Name"] = cand.Short_form; cand["Formula"] = cand.apply(C.formula, axis=1)
cand["Tolerance_factor"], cand["Octahedral_factor"] = Xc.tolerance_factor.values, Xc.octahedral_factor.values
fam_mean = raw.groupby("Perovskite_composition_short_form").JV_default_PCE.mean()
cand["DB_family_mean_PCE"] = cand.Name.map(fam_mean); cand["Gain_vs_family_mean"] = cand.Pred_PCE - cand.DB_family_mean_PCE
cand["List"] = np.where(cand.Novelty_status.str.startswith("already"), "in database",
                np.where(cand.Novelty_status.str.startswith("new ion-set"), "B", np.where(cand.N_DB_devices_same_ionset >= 100, "A", "other (known ion-set, <100 devices)")))
cand.drop(columns=["Perovskite_composition_inorganic", "Perovskite_composition_leadfree"] + list(C.CTX.keys())).to_csv(f"{OUT}/candidates.csv", index=False)
log(cand.Novelty_status.value_counts().to_string())

if STABONLY:
    FINAL["STAB"].save_model(f"{OUT}/model_STAB.json"); sj(STAB_CATS, "categories_stability.json")
    META.update(stability_features=list(Xs_all.columns), n_stability_rows=int(len(p3)), n_candidates=int(len(cand)), novelty_counts=cand.Novelty_status.value_counts().to_dict(),
                n_listA=int((cand.List == "A").sum()), n_listB=int((cand.List == "B").sum()), n_pareto=int(cand.Pareto_optimal.sum()), stab_only_refresh=True)
    sj(META, "meta.json"); log("DONE (stab-only)"); sys.exit(0)

# --- database index (pure rows, canonical ion order) ---
pure = np.where(C.pure_mask(X_all))[0]
dbi = pd.DataFrame(Mdb[pure], columns=[f"{s}_{k}" for s, k in C.VEC_IONS]); dbi["row"] = pure
dbi["Name"] = sf.values[pure]; dbi["PCE"] = Y["PCE"][pure]
dbi.to_csv(f"{OUT}/db_index.csv", index=False)
dbc = raw.groupby("Perovskite_composition_short_form").agg(N_devices=("JV_default_PCE", "size"), Mean_PCE=("JV_default_PCE", "mean"), Max_PCE=("JV_default_PCE", "max")).reset_index().sort_values("N_devices", ascending=False)
dbc.to_csv(f"{OUT}/E_database_compositions.csv", index=False)

# --- save models, meta, categories, reference predictions ---
for k, m in FINAL.items(): m.save_model(f"{OUT}/model_{k}.json")
sj(CATS, "categories.json"); sj(STAB_CATS, "categories_stability.json")
ref_idx = np.random.default_rng(0).choice(len(X_all), 200, replace=False)
refX = X_all.iloc[ref_idx]; refX.to_csv(f"{OUT}/ref_X.csv", index=False)
refP = pd.DataFrame({t: FINAL[t].predict(refX) for t in C.TARGETS}); refP.to_csv(f"{OUT}/ref_predictions.csv", index=False)
rs = resdf.set_index(["Target", "Split"])
err = {t: {sp: dict(MAE=float(rs.loc[(t, sp), "MAE"]), RMSE=float(rs.loc[(t, sp), "RMSE"])) for sp in ("random 80/20", "blocked 80/20")} for t in C.TARGETS}
ranges = {c: [float(X_all[c].min()), float(X_all[c].max())] for c in ("log10_area_cm2",)}
sj(dict(features=FEATURES, stability_features=list(Xs_all.columns), best_config=BEST, seed=C.SEED, novelty_tol=C.NOVELTY_TOL, w_pce=C.W_PCE,
        block_raw=C.BLOCK_RAW, block_stab=C.BLOCK_STAB, n_raw=len(raw), n_valid=int(len(idx_ok)), n_stability_rows=int(len(p3)),
        split_sizes=dict(random=[len(tr_r), len(te_r)], blocked=[len(tr_g), len(te_g)]), error=err, blocked_cv_mean=float(np.mean(cv)), blocked_cv_std=float(np.std(cv)),
        top_features=top5, ranges=ranges, smoke=SMOKE, ea_ev=C.EA_EV, t_ref_k=C.T_REF_K,
        n_candidates=int(len(cand)), novelty_counts=cand.Novelty_status.value_counts().to_dict(),
        n_listA=int((cand.List == "A").sum()), n_listB=int((cand.List == "B").sum()), runtime_min=(time.time() - T0) / 60), "meta.json")
log("DONE")
