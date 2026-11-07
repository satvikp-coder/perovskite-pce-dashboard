import os, sys, hmac
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import streamlit as st
from pv2 import core as C, store as ST

st.set_page_config(page_title="Perovskite solar cell predictor", layout="wide")

# ---------------------------------------------------------------- optional password gate (env APP_PASSWORD)
_pw = os.environ.get("APP_PASSWORD")
if _pw and not st.session_state.get("auth_ok"):
    st.title("Perovskite solar cell predictor")
    entered = st.text_input("Password", type="password", key="pw_in")
    if entered:
        if hmac.compare_digest(entered.encode(), _pw.encode()): st.session_state["auth_ok"] = True; st.rerun()
        else: st.error("Wrong password.")
    st.stop()


@st.cache_resource
def get_store(): return ST.load_store(os.environ.get("MODEL_STORE", ST.DEFAULT_STORE))
S = get_store(); meta = S["meta"]; cats = S["cats"]
ERR = meta["error"]["PCE"]; BAND = ERR["blocked 80/20"]["MAE"]

STAB_WARN = ("Stability warnings (Paper 3): (1) the database column is the TEST DURATION, not time-to-80%. (2) The data contain NO numeric temperature "
             "(only words: Constant/Uncontrolled/Cycled) and humidity for very few devices, so temperature/humidity enter only as a constant-multiplier SCENARIO "
             "and cannot be learnt. (3) TS80m = TS80 x A_T x A_RH x A_light with Ea = 0.6 eV, reference 300 K, 20% RH, 1 sun. (4) Scenarios do not change model R2 or ranking.")

PRESETS = {
    "Triple-cation (Cs/FA/MA, Pb, I/Br)": dict(a_Cs=0.05, a_FA=0.79, a_MA=0.16, a_Rb=0.0, b_Pb=1.0, b_Sn=0.0, x_I=0.83, x_Br=0.17, x_Cl=0.0),
    "FAPbI3": dict(a_Cs=0.0, a_FA=1.0, a_MA=0.0, a_Rb=0.0, b_Pb=1.0, b_Sn=0.0, x_I=1.0, x_Br=0.0, x_Cl=0.0),
    "MAPbI3 (classic)": dict(a_Cs=0.0, a_FA=0.0, a_MA=1.0, a_Rb=0.0, b_Pb=1.0, b_Sn=0.0, x_I=1.0, x_Br=0.0, x_Cl=0.0),
}
EXAMPLE = "Triple-cation (Cs/FA/MA, Pb, I/Br)"
for k, v in PRESETS[EXAMPLE].items(): st.session_state.setdefault(k, v)


def set_preset(name):
    for k, v in PRESETS[name].items(): st.session_state[k] = v
    st.session_state["example_used"] = (name == EXAMPLE)


def show_table(df, caption, details_df=None, key=None, **kw):
    """One-line caption + table; technical columns appear only after ticking 'show details'."""
    st.caption(caption)
    if details_df is not None and st.checkbox("Show details (technical columns)", key=f"det_{key}"): df = details_df
    st.dataframe(df, hide_index=True, **kw)


# ---------------------------------------------------------------- sidebar help
with st.sidebar:
    st.header("Help")
    st.markdown("""
1. Open the **Predict** tab (or press **Try an example**).
2. Pick a preset or type the A-site, B-site and halide fractions (each group must add up to 1).
3. Read the **result card**: the number, its error range and the traffic light.
4. Open **Best candidates** to see untested compositions that look promising.
5. Check **How accurate is it** before trusting any number.
""")
    st.caption("Estimates come from a statistical model of a literature database, not from experiments.")

# ---------------------------------------------------------------- start here
st.title("Perovskite solar cell predictor")
with st.container(border=True):
    st.subheader("Start here")
    l, r = st.columns([3, 1])
    l.markdown("- This tool estimates the efficiency (PCE) of a perovskite solar cell from its composition, using a model trained on a public literature database.\n"
               "- It also lists untested compositions and shows how accurate the model is.\n"
               "- Results are estimates with an error range, not measurements and not novelty claims.")
    r.button("Try an example", type="primary", use_container_width=True, on_click=set_preset, args=(EXAMPLE,), key="btn_example")
    if st.session_state.get("example_used"): st.success("Example loaded: the common triple-cation composition. See the result in the Predict tab below.")

tabs = st.tabs(["Predict", "Best candidates", "How accurate is it", "Why", "Stability scenarios", "About the data"])

# ---------------------------------------------------------------- 1 Predict
with tabs[0]:
    st.subheader("Predict a composition")
    p1, p2, p3_, _sp = st.columns([1, 1, 1, 2])
    for col, nm in zip((p1, p2, p3_), PRESETS):
        col.button(f"Preset: {nm.split(' (')[0]}", on_click=set_preset, args=(nm,), key=f"pre_{nm}", use_container_width=True)
    c1, c2, c3 = st.columns(3)
    def site(col, title, ions, key):
        col.markdown(f"**{title}** (fractions add up to 1)")
        return {i: col.number_input(i, 0.0, 1.0, step=0.01, format="%.2f", key=f"{key}_{i}") for i in ions}
    A = site(c1, "A-site cation", ["Cs", "FA", "MA", "Rb"], "a")
    B = site(c2, "B-site metal", ["Pb", "Sn"], "b")
    X = site(c3, "Halide", ["I", "Br", "Cl"], "x")
    with st.expander("Advanced (device architecture, additives, area, processing) - defaults are fine for a first look"):
        d1, d2, d3, d4 = st.columns(4)
        etl = d1.selectbox("Electron transport layer", [c for c in cats["etl"] if c not in ("None", "Unknown")], key="etl")
        htl = d2.selectbox("Hole transport layer", [c for c in cats["htl"] if c not in ("None", "Unknown")], key="htl")
        contact = d3.selectbox("Back contact", [c for c in cats["contact"] if c not in ("None", "Unknown")], key="contact")
        area = d4.number_input("Cell area (cm2)", 0.001, 100.0, 0.1, 0.01, format="%.3f", key="area")
        e1, e2, e3 = st.columns(3)
        dep = e1.selectbox("Deposition atmosphere", ["N2", "Air", "Ar", "Vacuum"], key="dep")
        jv = e2.selectbox("Test (JV) atmosphere", list(cats["jvatm"]), key="jv")
        adds = e3.multiselect("Additives", cats["additives"], key="adds")
    errs = [f"{n}: {m}" for n, s_ in (("A-site", A), ("B-site", B), ("X-site", X)) if (m := ST.validate_fractions(s_))]
    if errs:
        st.error("Invalid fractions - " + "; ".join(errs))
    else:
        p = ST.predict(S, ST.make_row(A, B, X, etl, htl, contact, dep, jv, adds, area))
        level, why = ST.reliability(p)
        icon = {"High": "GREEN", "Medium": "AMBER", "Low": "RED"}[level]
        lo, hi = max(p["PCE"] - BAND, 0.0), p["PCE"] + BAND
        with st.container(border=True):
            k1, k2 = st.columns([2, 1])
            k1.metric("Predicted efficiency (PCE)", f"{p['PCE']:.1f} %")
            k1.markdown(f"**Likely range:** {lo:.1f} to {hi:.1f} % (typical error +/- {BAND:.1f} percentage points on new studies)")
            k2.markdown(f"**Reliability: {level}** ({icon})")
            st.markdown({"High": ":green[High reliability]", "Medium": ":orange[Medium reliability]", "Low": ":red[Low reliability]"}[level] +
                        f" - {why} Predictions are compressed toward the database average, so very high or very low values are under-stated.")
        if p["extrapolation"] != "LOW": st.warning(f"Extrapolation warning: {p['extrapolation']}. The model has little or no data here; treat the number as a hypothesis.")
        if not p["plausible"]: st.warning("Chemistry check: outside the usual tolerance/octahedral-factor window (descriptors only, not proof of stability).")
        if p["novelty"].startswith("already"): st.info(f"Database status: already in database (ratio within {C.NOVELTY_TOL} of a tested composition).")
        else: st.info(f"Database status: not in database ({p['novelty']}). This is not a novelty claim; no literature search has been done.")
        with st.expander("Show details (Voc, Jsc, FF, physics check, nearest compositions)"):
            m = st.columns(5)
            m[0].metric("PCE (direct model)", f"{p['PCE']:.2f} %"); m[1].metric("Voc", f"{p['Voc']:.3f} V"); m[2].metric("Jsc", f"{p['Jsc']:.2f} mA/cm2")
            m[3].metric("FF", f"{p['FF']:.3f}"); m[4].metric("Physics PCE (Voc x Jsc x FF)", f"{p['PCE_physics']:.2f} %")
            st.caption(f"Confidence = {p['Confidence']:.2f}: agreement between the direct PCE model and the physics route. Error range uses blocked-split MAE {BAND:.2f} pp "
                       f"(random-split MAE {ERR['random 80/20']['MAE']:.2f} pp is optimistic); blocked RMSE {ERR['blocked 80/20']['RMSE']:.2f} pp.")
            st.caption(f"Tolerance factor {p['tolerance_factor']:.3f}, octahedral factor {p['octahedral_factor']:.3f} (windows {C.TOLERANCE_WINDOW} and {C.OCTAHEDRAL_WINDOW}); "
                       f"traffic light is a rule of thumb from confidence, extrapolation and these windows.")
            show_table(pd.DataFrame(p["nearest"]), "Closest compositions already in the database (distance = largest ion-fraction difference; smaller is closer).", key="near")

# ---------------------------------------------------------------- 2 Best candidates
with tabs[1]:
    st.subheader("Best candidates: compositions that are not in the database")
    st.caption("List A = untested ratios of families with at least 100 devices. List B = new ion-sets (never in database; high risk, extrapolation). "
               "Score = (0.7 x PCE_norm + 0.3 x stability_norm) x confidence. The stability axis is a weak proxy (test duration).")
    cand = S["candidates"]; lst = st.radio("List", ["A", "B"], horizontal=True, key="cand_list")
    pool = cand[cand.List == lst]
    g1, g2, g3, g4 = st.columns(4)
    min_pce = g1.slider("Min predicted PCE (%)", 0.0, float(max(pool.Pred_PCE.max(), 1)), 0.0, key="f_pce") if len(pool) else 0
    min_conf = g2.slider("Min confidence", 0.0, 1.0, 0.0, 0.05, key="f_conf")
    a_sets = sorted(pool.Perovskite_composition_a_ions.unique()); sel_a = g3.multiselect("A-site ion set", a_sets, key="f_a")
    only_pareto = g4.checkbox("Pareto-optimal only", key="f_par")
    f = pool[(pool.Pred_PCE >= min_pce) & (pool.Confidence >= min_conf)]
    if sel_a: f = f[f.Perovskite_composition_a_ions.isin(sel_a)]
    if only_pareto: f = f[f.Pareto_optimal]
    f = f.sort_values("Score", ascending=False)
    f = f.assign(Significance=np.where(f.Gain_vs_family_mean.abs() < BAND, "not significant (gain < model error)", "above model error"))
    st.write(f"{len(f):,} candidates shown of {len(pool):,} in list {lst}. Model error used for significance: blocked MAE {BAND:.2f} pp. "
             "Differences smaller than this are NOT significant.")
    simple = ["Formula", "Pred_PCE", "Gain_vs_family_mean", "Significance", "Confidence", "Extrapolation_flag"]
    full = ["Name", "Formula", "Pred_PCE", "Pred_Voc_V", "Pred_Jsc_mAcm2", "Pred_FF", "Pred_stability_proxy_h", "Confidence", "Score", "Gain_vs_family_mean",
            "Significance", "Extrapolation_flag", "Novelty_status", "N_DB_devices_same_ionset", "Pareto_optimal"]
    ren = {"Pred_PCE": "Predicted PCE (%)", "Gain_vs_family_mean": "Gain vs similar known compositions (pp)", "Extrapolation_flag": "Extrapolation risk"}
    show_table(f[simple].head(200).rename(columns=ren).round(3),
               "Each row is one untested composition, best first; compare 'Gain' with the model error above before getting excited. Top 200 shown.",
               f[full].head(200).round(3), key="cand")
    st.download_button("Download filtered list (CSV)", f[full].to_csv(index=False).encode(), f"candidates_list_{lst}.csv", "text/csv", key="dl_cand")
    if st.checkbox("Show PCE vs stability chart", key="cand_chart"):
        import plotly.graph_objects as go
        bg = cand.sample(min(4000, len(cand)), random_state=0)
        fig = go.Figure(); fig.add_scatter(x=bg.Pred_PCE, y=bg.Pred_stability_proxy_h, mode="markers", marker=dict(size=3, color="lightgray"), name="all candidates (sample)")
        pf = cand[cand.Pareto_optimal].sort_values("Pred_PCE"); fig.add_scatter(x=pf.Pred_PCE, y=pf.Pred_stability_proxy_h, mode="lines+markers", name="Pareto front", line=dict(color="#0C447C"))
        fig.add_scatter(x=f.head(20).Pred_PCE, y=f.head(20).Pred_stability_proxy_h, mode="markers", name=f"top 20 of filtered list {lst}", text=f.head(20).Formula, marker=dict(size=9, color="#D85A30"))
        fig.update_layout(xaxis_title="predicted PCE (%)", yaxis_title="stability proxy (predicted test hours)", height=450)
        st.caption("Each dot is a candidate; up and to the right is better on both axes. The vertical axis is test duration, not lifetime.")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 3 How accurate is it
with tabs[2]:
    import plotly.express as px
    st.subheader("How accurate is it?")
    st.info("Random 80/20 (seed 42) is optimistic: neighbouring rows come from the same studies. Blocked split (consecutive-row blocks of 200) is stricter and is the better guide for new compositions. "
            "Ref_ID is a row counter, not a paper ID, so no split here is paper-wise. Predictions are compressed toward the mean "
            f"(blocked slope of predicted vs actual PCE = {S['shrinkage']['slope_pred_vs_actual_blocked']:.2f}).")
    mt = S["metrics_targets"].round(3)
    show_table(mt[["Target", "Split", "R2", "MAE"]], "R2: 1 is perfect, 0 is no better than guessing the average. MAE: typical size of the error (PCE in percentage points, Voc in V, Jsc in mA/cm2).",
               mt[["Target", "Split", "R2", "R_pearson", "MAE", "RMSE", "MAPE_pct", "N_test"]], key="mt")
    cv = S["cv_blocked_pce"]
    st.write(f"Blocked {len(cv)}-fold cross-validation (PCE): R2 = {cv.R2.mean():.3f} +/- {cv.R2.std(ddof=0):.3f}")
    with st.expander("Show details: model comparison, cross-validation folds, leave-family-out"):
        if S.get("model_comparison") is not None:
            mc = S["model_comparison"].round(3)
            for sp_ in ("random 80/20", "blocked 80/20"): show_table(mc[mc.Split == sp_].drop(columns="Split"), f"PCE model comparison, {sp_} (same splits; tuned XGBoost is the deployed model; higher R2 and lower MAE are better).", key=f"mc{sp_[:3]}")
        show_table(cv.round(3), "R2 of each cross-validation fold (each fold tests on study blocks the model did not see).", key="cv")
        show_table(S["leave_family_out"].round(3), "Leave-family-out for PCE: the model never saw this composition family while training; lower R2 means weaker extrapolation.", key="lfo")
    q1, q2 = st.columns(2)
    tgt = q1.selectbox("Target", C.TARGETS, key="q_t"); spl = q2.radio("Split", ["random", "blocked"], horizontal=True, key="q_s")
    tp = S[f"test_predictions_{spl}"]; fig = px.scatter(tp.sample(min(4000, len(tp)), random_state=0), x=f"actual_{tgt}", y=f"pred_{tgt}", opacity=0.3)
    lo, hi = float(tp[f"actual_{tgt}"].min()), float(tp[f"actual_{tgt}"].max()); fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x", line=dict(dash="dash", color="black"))
    fig.update_layout(title=f"{tgt} predicted vs measured, {spl} split", height=450)
    st.caption("Each dot is a test device; dots on the dashed line would be perfect predictions. A flatter cloud shows the compression toward the mean.")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- 4 Why
with tabs[3]:
    import plotly.express as px
    st.subheader("Why does the model predict what it predicts?")
    st.caption("SHAP values of the PCE model (random-split training set, 2000 test rows). Associations in the data, not causal effects. Voc, Jsc and FF are never inputs.")
    sh = S["C_shap_all_features_pce"]; n = st.slider("Features shown", 5, 40, 20, key="shap_n")
    top = sh.head(n).iloc[::-1]; fig = px.bar(top, x="Mean_abs_SHAP", y="Feature", orientation="h", hover_data=["Direction"], height=max(350, 18 * n))
    st.caption("Longer bar = that input moves the predicted PCE more, on average (percentage points).")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Show details: dependence plots and full table"):
        feats = S["shap_features"]; names = list(sh.Feature.head(12)); sel = st.selectbox("Feature for SHAP dependence", names, key="shap_f")
        j = feats.index(sel); fig = px.scatter(x=S["shap_X"][:, j], y=S["shap"][:, j], labels={"x": sel, "y": "SHAP value (pp PCE)"}, opacity=0.4)
        st.caption("Each dot is a device; height shows how much this input pushed its predicted PCE up or down.")
        st.plotly_chart(fig, use_container_width=True)
        pdp = S["pdp_top5"]; pf_ = st.selectbox("Partial dependence feature", list(pdp.Feature.unique()), key="pdp_f")
        d = pdp[pdp.Feature == pf_]; fig = px.line(d, x="value", y="predicted_PCE", markers=True, labels={"value": pf_, "predicted_PCE": "predicted PCE (%)"})
        st.caption("Average predicted PCE as this one input is varied and everything else is left as in the data.")
        st.plotly_chart(fig, use_container_width=True)
        show_table(sh.round(4), "All inputs ranked by mean absolute SHAP value (importance).", key="shap")

# ---------------------------------------------------------------- 5 Stability scenarios
with tabs[4]:
    st.subheader("Stability scenarios (constant-multiplier what-if)")
    st.warning(STAB_WARN)
    cc = S["candidates"]; options = cc[cc.List == "A"].sort_values("Score", ascending=False).head(50).Formula.tolist() or cc.Formula.head(50).tolist()
    pick = st.selectbox("Composition (top List A candidates)", options, key="s_comp")
    base = float(cc[cc.Formula == pick].Pred_stability_proxy_h.iloc[0])
    s1, s2, s3 = st.columns(3)
    T = s1.slider("Temperature (deg C) - scenario only", 25, 200, 50, key="s_T"); RH = s2.slider("Relative humidity (%)", 0, 100, 50, key="s_RH")
    L = s3.slider("Light intensity (mW/cm2)", 0, 200, 100, key="s_L")
    aT = float(C.arrhenius(T + 273.15)); aR = float(C.rh_factor(RH)); aL = float(C.light_factor(L)) if L > 0 else 1.0
    r = st.columns(5)
    r[0].metric("Proxy (test duration, h)", f"{base:,.0f}"); r[1].metric("A_T", f"{aT:.2f}"); r[2].metric("A_RH", f"{aR:.2f}"); r[3].metric("A_light", f"{aL:.2f}")
    r[4].metric("Scenario TS80m-style value (h)", f"{base * aT * aR * aL:,.0f}")
    st.caption("Value = proxy x A_T x A_RH x A_light. It multiplies a test duration, so it is NOT a measured or predicted T80/TS80. A_light = 1 for dark/zero light (v1 convention).")
    sc = S["stability_scenarios"].round(3)
    show_table(sc, "Pre-computed scenarios from the training pipeline: each row multiplies the stability proxy by the stated factors; the model itself is unchanged.", key="scen")
    show_table(S["stability_ablation"].round(3), "What the stability model learns: R2 on log hours when feature groups are removed (higher = more informative group).", key="abl")

# ---------------------------------------------------------------- 6 About the data
with tabs[5]:
    st.subheader("About the data")
    sf_ = S["stab_facts"]
    st.markdown(f"""
**Dataset.** Perovskite Database extract: {meta['n_raw']:,} devices x 34 columns; {meta['n_valid']:,} rows with valid PCE, Voc, Jsc, FF used for training.
Stability subset: {meta['n_stability_rows']:,} rows with a test duration (median {sf_['median_test_duration_h']:.0f} h); numeric temperature values: {sf_['n_numeric_temperature_values']}; rows with any humidity: {sf_['n_rows_with_any_humidity']}.

**Leakage rules.** PCE models never use Voc, Jsc or FF as inputs (PCE = Voc x Jsc x FF at 1 sun); they are separate targets. Band gap and thickness are excluded.
**Splits.** Random 80/20 (seed 42) and blocked split (consecutive-row blocks: {meta['block_raw']} raw, {meta['block_stab']} stability). Ref_ID is a row counter, so no split is paper-wise.
Many test rows have an identical input vector to a training row (same composition, stack and conditions from other devices), which makes the random split optimistic.

**Caveats.**
- Scope is Pb-based perovskites; Sn only in small fractions in training; Ge and Bi candidates are switched off.
- Inorganic A-site means Cs or Rb only; FA and MA are organic/hybrid.
- 'Not in database' = whole database, canonical ion order (Br before I), ratio tolerance {meta['novelty_tol']}; no literature search done, so it is not a novelty claim.
- Predictions are compressed toward the mean; top candidate gains are often below model error.
- Stability target is test duration, a weak proxy; chemistry window removes very few candidates.
- Tolerance and octahedral factors use approximate radii and are descriptors only.

**Open decisions.**
1. Lead-free / Ge / Bi: include or defer (data are very sparse for these elements).
2. How to report temperature and RH (scenario multipliers vs measured columns).
3. Stability-target redesign (test duration is not time-to-80%).
4. Request to the database owners for T80/TS80 columns and a study-ID / DOI column (for true study-wise validation).
""")
