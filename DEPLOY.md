# Deploying to Render (free tier)

The app reads only `model_store/` at runtime (no raw data, no retraining). Repo size without ignored folders is about 66 MB (largest file `model_store/candidates.csv`); nothing exceeds 100 MB. Measured peak RAM with the store loaded: about 285 MB.

1. The repository is public; it contains no raw data (`.gitignore` excludes raw CSVs, `runs/`, `data_derived/`, `results_v2_ref/`).
2. To publish your own copy: `git init`, `git add .`, `git commit`, `git branch -M main`, then create the GitHub repository and push.
3. In Render: **New > Blueprint**, connect the GitHub account, pick the repository (`perovskite-pce-dashboard`). Render reads `render.yaml` (build `pip install -r requirements.txt`, start `streamlit run dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`).
4. When prompted for `APP_PASSWORD`, enter a password (or set it later under Environment). If it is set, the app asks for it before showing anything; if empty/unset the app is public.
5. Wait for the first deploy, then open the `https://<name>.onrender.com` URL and enter the password.
6. Free tier: the service sleeps after ~15 minutes without traffic; the first request afterwards takes about 30-60 s to wake. RAM limit on free tier is 512 MB (the app needs about 285 MB).

Local check with the same command: `python -m streamlit run dashboard/app.py --server.port 10000 --server.address 0.0.0.0 --server.headless true`, then open http://localhost:10000.
Training/tests need `pip install -r requirements-dev.txt`.
