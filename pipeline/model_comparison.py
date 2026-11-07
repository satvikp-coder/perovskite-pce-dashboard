"""PCE model comparison (Ridge, Linear, RF, XGBoost default/tuned, LightGBM) on the same random and blocked splits as train_export.py.
Usage: python pipeline/model_comparison.py --raw ../merged_perovskite.csv  -> model_store/results/model_comparison_pce.csv"""
import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from pv2 import core as C
ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_store"))
a = ap.parse_args()
raw = C.read_csv_safe(a.raw).reset_index(drop=True); X, _ = C.build_features(raw); G = C.group_key(raw, C.BLOCK_RAW); ok = C.valid_rows(raw); y = raw.JV_default_PCE.values
BEST = json.load(open(f"{a.out}/meta.json"))["best_config"]; S = C.SEED
imp = lambda m: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)
models = {"Linear Regression": lambda: imp(LinearRegression()), "Ridge Regression": lambda: imp(Ridge(alpha=1.0)),
          "Random Forest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=S)),
          "XGBoost (default settings)": lambda: XGBRegressor(random_state=S, n_jobs=-1, tree_method="hist", verbosity=0),
          "LightGBM (default settings)": lambda: LGBMRegressor(random_state=S, n_jobs=-1, verbose=-1),
          "XGBoost (tuned)": lambda: C.make_xgb(**BEST)}
rows = []
for sp, (tr, te) in {"random 80/20": C.random_split(ok), "blocked 80/20": C.blocked_split(ok, G)}.items():
    for name, f in models.items():
        t = time.time(); p = f().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]); m = C.metrics(y[te], p, min_y=1.0)
        rows.append(dict(Model=name, Split=sp, **m, Seconds=round(time.time() - t, 1))); print(rows[-1], flush=True)
os.makedirs(f"{a.out}/results", exist_ok=True); pd.DataFrame(rows).to_csv(f"{a.out}/results/model_comparison_pce.csv", index=False)
