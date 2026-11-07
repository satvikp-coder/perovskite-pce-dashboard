"""Core logic of perovskite pipeline v2 (moved from artifacts/code_v2/perovskite_pipeline_v2.py without logic changes)."""
import re, itertools
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

SEED = 42
NOVELTY_TOL = 0.03
INCLUDE_GE = False
EA_EV, KB_EV, T_REF_K = 0.6, 8.617e-5, 300.0
BLOCK_RAW, BLOCK_STAB = 200, 100
W_PCE = 0.7
TARGETS = ["PCE", "Voc", "Jsc", "FF"]
UNITS = {"PCE": "%", "Voc": "V", "Jsc": "mA/cm2", "FF": "(fraction)"}
INORGANIC_A = {"Cs", "Rb"}   # inorganic A-site means Cs or Rb only; FA and MA are organic/hybrid
TOLERANCE_WINDOW, OCTAHEDRAL_WINDOW = (0.80, 1.05), (0.40, 0.90)

RADII_A = {"Cs": 1.88, "Rb": 1.72, "K": 1.64, "MA": 2.17, "FA": 2.53, "GA": 2.78}
RADII_B = {"Pb": 1.19, "Sn": 1.10, "Ge": 0.73, "Bi": 1.03, "Sb": 0.76}
RADII_X = {"I": 2.20, "Br": 1.96, "Cl": 1.81}
LIST_A, LIST_B, LIST_X = ["Cs", "FA", "MA", "Rb", "GA"], ["Pb", "Sn", "Bi", "Sb", "Ge"], ["I", "Br", "Cl"]
ALIAS = {"Gua": "GA", "Guanidinium": "GA", "G": "GA"}
VEC_IONS = [("A", k) for k in ["Cs", "FA", "MA", "Rb"]] + [("B", k) for k in ["Pb", "Sn", "Ge"]] + [("X", k) for k in ["Br", "Cl", "I"]]

CTX = dict(Cell_stack_sequence="SLG | FTO | TiO2-c | TiO2-mp | Perovskite | Spiro-MeOTAD | Au", Substrate_stack_sequence="SLG | FTO",
           Perovskite_additives_compounds="Undoped", Perovskite_deposition_synthesis_atmosphere="N2", JV_test_atmosphere="Air",
           Cell_area_measured=0.1, JV_light_intensity=100.0, Stability_atmosphere="Air", Stability_light_intensity=100.0)


def read_csv_safe(path):
    """Read a CSV trying utf-8, latin-1, cp1252; strip column names."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            d = pd.read_csv(path, low_memory=False, encoding=enc); d.columns = d.columns.str.strip(); return d
        except Exception: continue
    raise IOError(f"cannot read {path}")


# ---------------- physics ----------------
def pce_physics(voc, jsc, ff):
    """PCE (%) = Voc (V) x Jsc (mA/cm2) x FF at 1 sun (100 mW/cm2)."""
    return np.asarray(voc, float) * np.asarray(jsc, float) * np.asarray(ff, float)

def tolerance_factor(rA, rB, rX):
    """Goldschmidt tolerance factor from ionic radii of A, B, X sites."""
    return (rA + rX) / (np.sqrt(2) * (rB + rX))      # Goldschmidt (1926)

def octahedral_factor(rB, rX):
    """Octahedral factor rB/rX."""
    return rB / rX                                    # Li et al. (2008)

def arrhenius(T_K, ea=EA_EV, tref=T_REF_K):
    """Arrhenius temperature acceleration factor relative to reference temperature (K)."""
    return np.exp(-ea / KB_EV * (1.0 / np.asarray(T_K, float) - 1.0 / tref))

def rh_factor(rh):
    """Humidity factor: RH/20 above 20 % RH, otherwise 1."""
    rh = np.asarray(rh, float); return np.where(rh > 20, rh / 20.0, 1.0)

def light_factor(light_mWcm2):
    """Light intensity relative to 100 mW/cm2."""
    return np.asarray(light_mWcm2, float) / 100.0


# ---------------- features ----------------
def parse_site(ions, coeffs):
    if not isinstance(ions, str): return {}
    toks = [ALIAS.get(t.strip(), t.strip()) for t in ions.split("|")[0].split(";") if t.strip()]
    if not toks: return {}
    vals = []
    if isinstance(coeffs, str):
        for c in coeffs.split("|")[0].split(";"):
            c = c.strip()
            if c == "": continue
            try: vals.append(float(c))
            except ValueError: vals.append(np.nan)
    if len(vals) != len(toks) or any(np.isnan(vals)) or sum(vals) <= 0: vals = [1.0] * len(toks)
    tot = sum(vals); out = {}
    for t, v in zip(toks, vals): out[t] = out.get(t, 0.0) + v / tot
    return out

def mean_radius(parsed, radii):
    out = []
    for d in parsed:
        known = sum(v for k, v in d.items() if k in radii)
        out.append(sum(v * radii[k] for k, v in d.items() if k in radii) / known if known >= 0.9 else np.nan)
    return np.array(out)

def parse_stack(s):
    if not isinstance(s, str): return ("Unknown", "Unknown", "Unknown", np.nan, 0)
    toks = [t.strip() for t in s.split("|")]
    idx = [i for i, t in enumerate(toks) if t.lower().startswith("perovskite")]
    if not idx: return ("Unknown", "Unknown", "Unknown", len(toks), 0)
    i = idx[0]; before, after = toks[:i], toks[i+1:]
    meso = int(any(re.search(r"-mp\b", t.lower()) for t in toks))
    return (before[-1] if before else "None", after[0] if after else "None", after[-1] if after else "None", len(toks), meso)

def onehot(series, cats, prefix):
    out = {f"{prefix}_{c}": (series == c).astype(int).values for c in cats}
    out[f"{prefix}_other"] = (~series.isin(cats)).astype(int).values
    return out

def build_features(df, cats=None, stability=False):
    """Raw database-style rows -> numeric feature table. Never uses Voc/Jsc/FF. Pass learnt `cats` to featurise new rows."""
    fit = cats is None; cats = {} if fit else cats
    df = df.reset_index(drop=True); F = {}
    pa = [parse_site(i, c) for i, c in zip(df["Perovskite_composition_a_ions"], df["Perovskite_composition_a_ions_coefficients"])]
    pb = [parse_site(i, c) for i, c in zip(df["Perovskite_composition_b_ions"], df["Perovskite_composition_b_ions_coefficients"])]
    px = [parse_site(i, c) for i, c in zip(df["Perovskite_composition_c_ions"], df["Perovskite_composition_c_ions_coefficients"])]
    for parsed, listed, tag in [(pa, LIST_A, "xA"), (pb, LIST_B, "xB"), (px, LIST_X, "xX")]:
        tot = np.zeros(len(df))
        for k in listed:
            v = np.array([d.get(k, 0.0) for d in parsed]); F[f"{tag}_{k}"] = v; tot += v
        F[f"{tag}_other"] = np.clip(1 - tot, 0, 1)
        F[f"n{tag[1]}_ions"] = np.array([len(d) for d in parsed])
    rA, rB, rX = mean_radius(pa, RADII_A), mean_radius(pb, RADII_B), mean_radius(px, RADII_X)
    F["radius_A"], F["radius_B"], F["radius_X"] = rA, rB, rX
    F["tolerance_factor"] = tolerance_factor(rA, rB, rX)
    F["octahedral_factor"] = octahedral_factor(rB, rX)
    F["inorganic"] = df["Perovskite_composition_inorganic"].astype(str).str.lower().eq("true").astype(int).values
    F["leadfree"] = df["Perovskite_composition_leadfree"].astype(str).str.lower().eq("true").astype(int).values
    st = pd.DataFrame([parse_stack(s) for s in df["Cell_stack_sequence"]], columns=["etl", "htl", "contact", "n_layers", "meso"])
    sub = df["Substrate_stack_sequence"].fillna("Unknown")
    if fit:
        cats["etl"] = st.etl.value_counts().head(10).index.tolist(); cats["htl"] = st.htl.value_counts().head(10).index.tolist()
        cats["contact"] = st.contact.value_counts().head(6).index.tolist(); cats["substrate"] = sub.value_counts().head(4).index.tolist()
    F.update(onehot(st.etl, cats["etl"], "etl")); F.update(onehot(st.htl, cats["htl"], "htl"))
    F.update(onehot(st.contact, cats["contact"], "contact")); F.update(onehot(sub, cats["substrate"], "substrate"))
    F["n_layers"], F["mesoporous"] = st.n_layers.values, st.meso.values
    F["log10_area_cm2"] = np.log10(pd.to_numeric(df["Cell_area_measured"], errors="coerce").clip(lower=1e-4)).values
    F["jv_light_mWcm2"] = pd.to_numeric(df["JV_light_intensity"], errors="coerce").values
    dep = df["Perovskite_deposition_synthesis_atmosphere"].fillna("Unknown").astype(str)
    dep1 = dep.str.split(">>").str[0].str.strip()
    F.update(onehot(dep1, ["N2", "Air", "Ar", "Vacuum", "Unknown"], "dep")); F["dep_multistep"] = dep.str.contains(">>").astype(int).values
    jv = df["JV_test_atmosphere"].fillna("Unknown").astype(str)
    if fit: cats["jvatm"] = jv.value_counts().head(6).index.tolist()
    F.update(onehot(jv, cats["jvatm"], "jvatm"))
    toks = df["Perovskite_additives_compounds"].apply(lambda s: set(t.strip() for t in str(s).split(";")) if isinstance(s, str) else set())
    if fit:
        cnt = pd.Series([t for s in toks for t in s if t not in ("Undoped", "Unknown", "")]).value_counts()
        cats["additives"] = cnt.head(8).index.tolist()
    F["has_additive"] = toks.apply(lambda s: int(len(s - {"Undoped", "Unknown", ""}) > 0)).values
    for a in cats["additives"]: F[f"add_{a}"] = toks.apply(lambda s, a=a: int(a in s)).values
    if stability:
        sa = df["Stability_atmosphere"].fillna("Unknown").astype(str)
        if fit: cats["stabatm"] = sa.value_counts().head(6).index.tolist()
        F.update(onehot(sa, cats["stabatm"], "stabatm"))
        sl = pd.to_numeric(df["Stability_light_intensity"], errors="coerce")
        F["stab_light_mWcm2"] = sl.values; F["stab_dark"] = (sl == 0).astype(int).values
    X = pd.DataFrame(F)
    X.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", c) for c in X.columns]
    return (X, cats) if fit else X

def assert_no_leakage(columns):
    """PCE inputs must not contain Voc, Jsc, FF (or PCE itself)."""
    bad = [c for c in columns if re.search(r"(^|_)(voc|jsc|ff)(_|$)", c, re.I) or "default_pce" in c.lower() or c.lower() == "pce"]
    assert not bad, f"leakage columns in PCE inputs: {bad}"


# ---------------- splits / metrics / models ----------------
def group_key(df, block):
    """Ref_ID is a row counter, not a paper ID. Blocks of consecutive rows approximate study-wise grouping (NOT paper-wise)."""
    n = len(df) if hasattr(df, "__len__") else int(df)
    return np.arange(n) // block

def valid_rows(raw):
    """Return indices of rows whose PCE, Voc, Jsc and FF lie in plausible ranges."""
    ok = (raw.JV_default_PCE.between(0.01, 33) & raw.JV_default_Voc.between(0.2, 3.5) &
          raw.JV_default_Jsc.between(0.01, 50) & raw.JV_default_FF.between(0.1, 1.0))
    return np.where(ok.values)[0]

def random_split(idx_ok, seed=SEED):
    """Random 80/20 split of row indices."""
    return train_test_split(idx_ok, test_size=0.2, random_state=seed)

def blocked_split(idx_ok, groups, seed=SEED):
    """80/20 split keeping blocks of consecutive rows together (group-wise)."""
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(idx_ok, groups=groups[idx_ok]))
    return idx_ok[a], idx_ok[b]

def metrics(y, p, min_y=None):
    """R2, Pearson r, MAE, RMSE, MAPE (%) and test size for predictions p against y."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.abs(y) >= (min_y if min_y is not None else np.inf) if min_y is not None else np.ones(len(y), bool)
    return dict(R2=r2_score(y, p), R_pearson=float(np.corrcoef(y, p)[0, 1]), MAE=mean_absolute_error(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAPE_pct=float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100) if m.any() else np.nan, N_test=len(y))

def make_xgb(smoke=False, **kw):
    """Build the XGBoost regressor used throughout (smoke=True gives a tiny fast model)."""
    P = dict(n_estimators=400, max_depth=8, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
             tree_method="hist", max_bin=128, n_jobs=-1, random_state=SEED, verbosity=0)
    if smoke: P.update(n_estimators=40)
    P.update(kw); return XGBRegressor(**P)

CONFIGS = [dict(max_depth=6, learning_rate=0.10, n_estimators=300), dict(max_depth=8, learning_rate=0.08, n_estimators=400),
           dict(max_depth=10, learning_rate=0.05, n_estimators=500, min_child_weight=5),
           dict(max_depth=8, learning_rate=0.05, n_estimators=600, colsample_bytree=0.6)]


# ---------------- candidate grid ----------------
def simplex(n, step=0.05, minv=0.05, cap=None, rng=None):
    """Enumerate composition fractions on a grid that sum to 1 for n components."""
    k, mn = int(round(1 / step)), int(round(minv / step)); res = []
    def rec(prefix, rem, slots):
        if slots == 1:
            if rem >= mn: res.append(prefix + [rem])
            return
        for v in range(mn, rem - mn * (slots - 1) + 1): rec(prefix + [v], rem - v, slots - 1)
    rec([], k, n); res = [[x * step for x in r] for r in res]
    if cap and len(res) > cap: res = [res[i] for i in sorted(rng.choice(len(res), cap, replace=False))]
    return res

def fmt(vals, scale=1.0): return "; ".join(f"{v * scale:.2f}".rstrip("0").rstrip(".") for v in vals)

def candidate_options(include_ge=INCLUDE_GE, seed=SEED, smoke=False):
    rng = np.random.default_rng(seed)
    A_ION_SETS = [s for n in range(1, 5) for s in itertools.combinations(["Cs", "FA", "MA", "Rb"], n)]
    A_OPTS = []
    for s in A_ION_SETS:
        caps = {1: None, 2: None, 3: 60, 4: 120}[len(s)]
        grid = [[1.0]] if len(s) == 1 else simplex(len(s), 0.05, 0.05, caps, rng)
        for g in grid:
            if "Rb" in s and len(s) > 1 and g[s.index("Rb")] > 0.20: continue
            A_OPTS.append((s, g))
    B_OPTS = [(("Pb",), [1.0]), (("Sn",), [1.0])] + [(("Pb", "Sn"), [1 - x, x]) for x in (0.1, 0.25, 0.5, 0.75, 0.9)]
    if include_ge: B_OPTS += [(("Ge",), [1.0]), (("Pb", "Ge"), [0.9, 0.1]), (("Pb", "Ge"), [0.75, 0.25])]
    X_OPTS = [(("I",), [1.0]), (("Br",), [1.0]), (("Cl",), [1.0])]
    X_OPTS += [(("Br", "I"), [x, 1 - x]) for x in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    X_OPTS += [(("Cl", "I"), [x, 1 - x]) for x in (0.05, 0.1, 0.2, 0.3)] + [(("Br", "Cl"), [1 - x, x]) for x in (0.1, 0.3, 0.5)]
    X_OPTS += [(("Br", "Cl", "I"), [b_, c_, 1 - b_ - c_]) for b_, c_ in ((0.15, 0.05), (0.2, 0.1), (0.3, 0.1), (0.1, 0.1))]
    if smoke: A_OPTS, B_OPTS, X_OPTS = A_OPTS[::12], B_OPTS[:4], X_OPTS[::3]
    return A_OPTS, B_OPTS, X_OPTS

def composition_row(sa, ga, sb, gb, sx, gx):
    """One database-style row for a composition (X coefficients scaled to 3 sites as in the database)."""
    return dict(Perovskite_composition_a_ions="; ".join(sa), Perovskite_composition_a_ions_coefficients=fmt(ga),
                Perovskite_composition_b_ions="; ".join(sb), Perovskite_composition_b_ions_coefficients=fmt(gb),
                Perovskite_composition_c_ions="; ".join(sx), Perovskite_composition_c_ions_coefficients=fmt(gx, 3.0),
                Perovskite_composition_inorganic=str(set(sa) <= INORGANIC_A), Perovskite_composition_leadfree=str("Pb" not in sb),
                Short_form="".join(sa) + "".join(sb) + "".join(sx))

def build_candidate_table(include_ge=INCLUDE_GE, seed=SEED, smoke=False):
    A, B, X = candidate_options(include_ge, seed, smoke)
    rows = [dict(composition_row(sa, ga, sb, gb, sx, gx), **CTX) for (sa, ga), (sb, gb), (sx, gx) in itertools.product(A, B, X)]
    return pd.DataFrame(rows)

def chemistry_filter(Xc):
    return (Xc.tolerance_factor.between(*TOLERANCE_WINDOW) & Xc.octahedral_factor.between(*OCTAHEDRAL_WINDOW)).values

def formula(r):
    def part(ions, co):
        out = []
        for i, c in zip([x.strip() for x in str(r[ions]).split(";")], [float(x) for x in str(r[co]).split(";")]):
            out.append(i if abs(c - 1) < 1e-9 else f"{i}{c:g}")
        return "".join(out)
    return (part("Perovskite_composition_a_ions", "Perovskite_composition_a_ions_coefficients") + part("Perovskite_composition_b_ions", "Perovskite_composition_b_ions_coefficients") +
            part("Perovskite_composition_c_ions", "Perovskite_composition_c_ions_coefficients"))


# ---------------- novelty ----------------
def vec_cols(X): return np.column_stack([X[f"x{s}_{k}"].values for s, k in VEC_IONS])
def ionset_key(M): return [tuple(np.where(r > 1e-6)[0]) for r in M]
def pure_mask(X):
    return (X[["xA_other", "xB_other", "xX_other"]].sum(1) < 1e-6).values & (X[["xA_GA", "xB_Bi", "xB_Sb"]].sum(1) < 1e-6).values

def build_db_index(X_all):
    """Whole database restricted to the 10 tracked ions (pure rows only): matrix + ion-set key -> row indices."""
    Mdb = vec_cols(X_all); pure = pure_mask(X_all)
    keys = pd.Series(ionset_key(Mdb)); by_key = {}
    for i in np.where(pure)[0]: by_key.setdefault(keys.iloc[i], []).append(i)
    return Mdb, by_key

def novelty_check(Mc, Mdb, by_key, tol=NOVELTY_TOL):
    """Mc: candidate vectors in CANONICAL ion order (VEC_IONS, i.e. Br before I). Returns (status, nearest distance, n same ion-set)."""
    status, dist, nsame = [], [], []
    for i, key in enumerate(ionset_key(Mc)):
        rows_db = by_key.get(key)
        if not rows_db: status.append("new ion-set (never in database)"); dist.append(np.nan); nsame.append(0); continue
        d = np.abs(Mdb[rows_db] - Mc[i]).max(1).min(); dist.append(d); nsame.append(len(rows_db))
        status.append("already in database (ratio within tolerance)" if d <= tol else "known ion-set, new ratio")
    return status, dist, nsame

def pareto_mask(a, b):
    """Boolean mask of points not dominated when maximizing both a and b."""
    order = np.lexsort((-b, -a)); best_b = -np.inf; keep = np.zeros(len(a), bool)
    for i in order:
        if b[i] > best_b: keep[i] = True; best_b = b[i]
    return keep

def confidence(direct, physics):
    """Physics-route confidence in [0,1]: 1 minus relative gap between direct and Voc*Jsc*FF PCE."""
    direct, physics = np.asarray(direct, float), np.asarray(physics, float)
    return np.clip(1 - np.abs(direct - physics) / np.maximum(np.maximum(np.abs(direct), np.abs(physics)), 1e-3), 0, 1)

def composite_score(pce, stab, conf, w=W_PCE):
    """Weighted min-max blend of PCE and stability, scaled by confidence."""
    pce, stab = np.asarray(pce, float), np.asarray(stab, float)
    pn = (pce - pce.min()) / (pce.max() - pce.min() + 1e-9); sn = (stab - stab.min()) / (stab.max() - stab.min() + 1e-9)
    return (w * pn + (1 - w) * sn) * conf
