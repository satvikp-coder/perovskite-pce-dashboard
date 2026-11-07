import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, pytest
from xgboost import XGBRegressor
from pv2 import core as C

STORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_store")


def test_pce_formula():
    assert C.pce_physics(1.1, 23.0, 0.80) == pytest.approx(20.24)       # Voc(V) x Jsc(mA/cm2) x FF = PCE(%) at 1 sun
    assert C.pce_physics(1.0, 20.0, 0.5) == pytest.approx(10.0)


def test_tolerance_and_octahedral_factor():
    # MAPbI3 with this project's radii: (2.17+2.20)/(sqrt2*(1.19+2.20))
    assert C.tolerance_factor(2.17, 1.19, 2.20) == pytest.approx(4.37 / (np.sqrt(2) * 3.39))
    assert C.octahedral_factor(1.19, 2.20) == pytest.approx(1.19 / 2.20)
    assert 0.8 < C.tolerance_factor(2.17, 1.19, 2.20) < 1.05


def _mini_db():
    rows = [("Cs; FA", "0.1; 0.9", "Pb", "1", "Br; I", "0.3; 2.7"), ("FA", "1", "Pb", "1", "I", "3")]
    df = pd.DataFrame(rows, columns=["Perovskite_composition_a_ions", "Perovskite_composition_a_ions_coefficients", "Perovskite_composition_b_ions",
                                     "Perovskite_composition_b_ions_coefficients", "Perovskite_composition_c_ions", "Perovskite_composition_c_ions_coefficients"])
    for c, v in dict(Perovskite_composition_inorganic="False", Perovskite_composition_leadfree="False", Cell_stack_sequence="SLG | FTO | Perovskite | Au",
                     Substrate_stack_sequence="SLG | FTO", Cell_area_measured=0.1, JV_light_intensity=100, Perovskite_deposition_synthesis_atmosphere="N2",
                     JV_test_atmosphere="Air", Perovskite_additives_compounds="Undoped").items(): df[c] = v
    return df


def test_canonical_order_novelty():
    db = _mini_db(); X, cats = C.build_features(db)
    Mdb, by_key = C.build_db_index(X)
    # candidate written in the other ion order (I; Br) with the same ratios must be recognised as already in database
    cand = db.iloc[[0]].copy()
    cand["Perovskite_composition_c_ions"] = "I; Br"; cand["Perovskite_composition_c_ions_coefficients"] = "2.7; 0.3"
    Xc = C.build_features(cand, cats=cats)
    st, dist, n = C.novelty_check(C.vec_cols(Xc), Mdb, by_key, C.NOVELTY_TOL)
    assert st[0].startswith("already") and dist[0] < 1e-9
    # ratio 0.03 tolerance: 0.02 away -> in database, 0.10 away -> new ratio, unseen ion-set -> new ion-set
    for br, expect in [(0.1 + 0.02, "already"), (0.1 + 0.10, "known")]:
        c2 = db.iloc[[0]].copy(); c2["Perovskite_composition_c_ions_coefficients"] = f"{br * 3}; {(1 - br) * 3}"
        assert C.novelty_check(C.vec_cols(C.build_features(c2, cats=cats)), Mdb, by_key)[0][0].startswith(expect)
    c3 = db.iloc[[1]].copy(); c3["Perovskite_composition_a_ions"], c3["Perovskite_composition_a_ions_coefficients"] = "MA", "1"
    assert C.novelty_check(C.vec_cols(C.build_features(c3, cats=cats)), Mdb, by_key)[0][0].startswith("new ion-set")


def test_inorganic_definition():
    assert C.INORGANIC_A == {"Cs", "Rb"}
    assert C.composition_row(("Cs", "Rb"), [0.9, 0.1], ("Pb",), [1.0], ("I",), [1.0])["Perovskite_composition_inorganic"] == "True"
    assert C.composition_row(("Cs", "FA"), [0.1, 0.9], ("Pb",), [1.0], ("I",), [1.0])["Perovskite_composition_inorganic"] == "False"


def test_no_leakage():
    X, _ = C.build_features(_mini_db())
    C.assert_no_leakage(X.columns)
    with pytest.raises(AssertionError): C.assert_no_leakage(list(X.columns) + ["Voc"])
    with pytest.raises(AssertionError): C.assert_no_leakage(list(X.columns) + ["JV_default_FF"])
    if os.path.exists(os.path.join(STORE, "meta.json")):
        meta = json.load(open(os.path.join(STORE, "meta.json")))
        C.assert_no_leakage(meta["features"])


def test_split_sizes():
    raw = pd.DataFrame(dict(JV_default_PCE=np.linspace(1, 25, 5000), JV_default_Voc=1.0, JV_default_Jsc=20.0, JV_default_FF=0.7))
    idx = C.valid_rows(raw); g = C.group_key(raw, C.BLOCK_RAW)
    tr, te = C.random_split(idx); assert len(te) == 1000 and len(tr) == 4000
    trg, teg = C.blocked_split(idx, g); assert len(trg) + len(teg) == 5000 and abs(len(teg) - 1000) <= 200
    assert not set(g[trg]) & set(g[teg])                       # no block shared between train and test
    assert len(set(C.group_key(100, C.BLOCK_STAB))) == 1 and C.BLOCK_RAW == 200 and C.BLOCK_STAB == 100


@pytest.mark.skipif(not os.path.exists(os.path.join(STORE, "meta.json")), reason="run pipeline/train_export.py first")
def test_reloaded_models_reproduce_predictions():
    X = pd.read_csv(os.path.join(STORE, "ref_X.csv")); ref = pd.read_csv(os.path.join(STORE, "ref_predictions.csv"))
    for t in C.TARGETS:
        m = XGBRegressor(); m.load_model(os.path.join(STORE, f"model_{t}.json"))
        np.testing.assert_allclose(m.predict(X), ref[t].values, rtol=1e-4, atol=1e-4)


def test_reliability_levels():
    from pv2 import store as ST
    base = dict(extrapolation="LOW", plausible=True, Confidence=0.95)
    assert ST.reliability(base)[0] == "High"
    assert ST.reliability({**base, "Confidence": 0.7})[0] == "Medium"
    assert ST.reliability({**base, "extrapolation": "MEDIUM (far from seen ratios)"})[0] == "Medium"
    assert ST.reliability({**base, "extrapolation": "HIGH (ion-set never seen)"})[0] == "Low"
    assert ST.reliability({**base, "plausible": False})[0] == "Low"
