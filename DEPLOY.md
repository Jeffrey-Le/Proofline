# Deploying Proofline

About an hour, most of it waiting. Run everything from the `proofline` folder with `.venv` active.

## 0. Before you start (decisions only you can make)

- [ ] `data/profile.yaml`: graduation month and MS status are how you want recruiters to see them.
- [ ] `data/bullets.yaml`: every bullet is OK to show recruiters; set `verified: false` on employer-confidential or unpublished work.
- [ ] `static/privacy.html`: replace `privacy@example.com` with a real address.

## 1. Pass locally with the real model (~15 min)

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q                                   # 33 passed
cp .env.example .env    # then put your key (from a project WITH billing) in .env
python -m scripts.load_bullets && python -m scripts.load_bullets --sample
uvicorn app:app --port 8000                           # leave running; new terminal for the next line
python -m scripts.smoke --url http://localhost:8000   # must end with "All checks passed."
```

If a quality check fails, open http://localhost:8000/?sample, run the sample posting, and read
what it said. Fix prompts before deploying; it's the same code in the cloud.

## 2. Google Cloud setup (~10 min, once)

1. https://console.cloud.google.com → **New project** (e.g. `proofline-demo-2026`). Note the project ID.
2. **Billing** → link the project to your billing account. Note the billing account ID
   (`gcloud billing accounts list`).
3. Install the gcloud CLI, then:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```
4. Your Gemini key should come from AI Studio with this project (or another billed one) selected.

## 3. Deploy (~15 min, mostly the index build)

```bash
PROJECT=proofline-demo-2026 BILLING=XXXXXX-XXXXXX-XXXXXX GEMINI_API_KEY=... ./deploy.sh
```

It ends by printing three things. Save them:
- the public URL
- the recruiter link (`/?for=...`): checks against your real resume
- your owner link (`/#token=...`): no limits, shows timings and cost. Keep it private.

## 4. Verify the live site

```bash
python -m scripts.smoke --url https://<your-url> --share <share-token>
```

Then by hand: open the public URL on your phone, try a real posting with the sample resume,
open the recruiter link in a private window, and find one request in
Cloud Console → Logging (search `proofline.request`) to see its timings and cost.

## Updating later

Code change: `gcloud run deploy proofline --source . --region us-west1` (keeps all settings).
Resume change: re-run `VECTOR_BACKEND=firestore python -m scripts.load_bullets`.
New recruiter link: rotate the share secret (command in `deploy.sh`), then redeploy.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Build fails with a permission error | Re-run `deploy.sh`; it grants the build role. Wait 2 min first. |
| `FAILED_PRECONDITION` / "index" in logs | The vector index is still building. Wait, retry. |
| Every check says "Proofline is busy" | Gemini 429: key is on the free tier or a model name changed. Check the key's project has billing; check model IDs in `proofline/config.py`. |
| "Something went wrong during the check" | Logs Explorer → filter `severity>=ERROR` for the stack trace. |
| Page loads, checks hang | Cloud Run timeout (120 s). Lower `MAX_SEARCHES` or check model latency. |
