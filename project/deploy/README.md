---
title: Phishing URL Triage
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Phishing URL Triage (Duke CYBERSEC 520 Final Project)

Gradio app that scores a URL as phishing or legitimate **from the URL text alone** using a Logistic Regression
and a Random Forest model. The URL is never visited. Advisory triage aid for a class project, not a full security product.

## Files
| File | Purpose |
|---|---|
| `app.py` | Gradio interface (URL input, model selector, threshold choice, score, flagged / not flagged, URL characteristics, inference time, examples) |
| `features.py` | URL-only feature extraction, shared with the training notebook |
| `models/` | `logistic_regression.joblib`, `random_forest.joblib`, `config.json` (frozen settings, thresholds, versions). Created by the notebook. |
| `requirements.txt` | Dependencies. The scikit-learn, pandas, numpy and joblib pins are written by the notebook from the training environment. |
| `Dockerfile` | Container definition (port 7860, non-root user). The Python version is set by the notebook, and the build runs both test files below. |
| `test_features.py` | Tests for URL normalization, input validation and feature extraction |
| `smoke_test.py` | End-to-end package check: versions match training, both models load and score, bad input is rejected cleanly |

## Configuration
- **Thresholds**: "Default" = 0.5. "Tuned" = threshold picked on validation data for the highest recall under a 1% false-positive rate (stored in `models/config.json`).
- **Environment variables** (optional): `MODEL_DIR` (model folder), `PORT` (default 7860).
- **Versions must match training.** `models/config.json` records the Python, scikit-learn, pandas, numpy and joblib versions that trained the models. The notebook writes matching pins to `requirements.txt` and the Python version to the `Dockerfile`. At startup the app compares them with what is installed and **refuses to start on a mismatch** (a saved model that loads without an error is not proven compatible). Set `ALLOW_VERSION_MISMATCH=1` only for debugging.
- **Input validation**: max 2048 characters, no spaces or control characters; only http/https URLs (scheme optional), real hostnames, valid IPv4/IPv6 addresses and numeric ports. Single-word hostnames such as `hello` or `localhost` are rejected.

## Test the package
```bash
cd deploy
python test_features.py   # URL normalization, validation, features
python smoke_test.py      # needs models/; checks versions, loading, scoring, bad input
```
The Docker build runs both, so a broken or mismatched package fails the build instead of failing after deployment.

## Run locally
```bash
pip install -r requirements.txt
python app.py            # open http://localhost:7860
```
Or with Docker:
```bash
docker build -t phishing-url-triage .
docker run -p 7860:7860 phishing-url-triage
```

## Deploy to Hugging Face Spaces
1. Run the training notebook so `deploy/models/` is filled.
2. Confirm `requirements.txt` and the Dockerfile's `FROM python:...` line were updated by the notebook (they should match `models/config.json`).
3. On huggingface.co: **New Space** → SDK **Docker** → blank template → create.
4. Upload the contents of this `deploy/` folder to the Space root (web upload, or `git clone` the Space repo, copy the files in, `git add . && git commit && git push`). Model files are small; if any file is over 10 MB use `git lfs track "*.joblib"` first.
5. The Space builds from the `Dockerfile` and serves on port 7860 (set by `app_port` in the header above). Check **Logs** if the build fails.
6. Build and test it in the same container Hugging Face will use. Locally: `docker build -t phishing-url-triage .` (this runs the tests), then `docker run -p 7860:7860 phishing-url-triage`. On Hugging Face, a failed build log shows which test failed. Then test it: paste a legitimate URL and a suspicious one, try both models, and confirm that bad input (empty, spaces) shows a message instead of crashing.

## Limitations
**Known limitation (important):** PhiUSIIL's legitimate URLs are mostly bare homepages, so the models partly learned that a URL with a path is phishing. On the independent secondary dataset both models flagged almost all legitimate URLs. The app is an advisory prototype and will over-flag ordinary URLs with long paths. See the notebook's Diagnosis section.

Only URL text is analyzed. Phishing hosted on legitimate domains or hidden behind link shorteners cannot be recognized this way, and the training data is historical. See the notebook for full evaluation on the primary (PhiUSIIL) and secondary (Kaggle Malicious URLs) datasets.
"# phishing_app" 
