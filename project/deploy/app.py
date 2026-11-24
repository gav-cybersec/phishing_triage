"""
Phishing URL Triage - Gradio app (CYBERSEC 520 final project).

Scores a URL using only its text. The URL is never visited or fetched.
Advisory triage aid, not a replacement for blocklists, gateways, or analyst review.
"""
import json
import os
import platform
import warnings
from pathlib import Path
import time

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from features import FEATURES, check_url_input, features_frame

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "models"))
MODEL_FILES = {
    "Random Forest": "random_forest.joblib",
    "Logistic Regression": "logistic_regression.joblib",
}

if not (MODEL_DIR / "config.json").exists():
    raise RuntimeError(f"No trained models found in {MODEL_DIR}. Run the notebook first so it saves them there.")

with open(MODEL_DIR / "config.json") as file:
    config = json.load(file)


def version_problems():
    """Compare the libraries used for training (stored in config.json) with the ones installed now."""
    saved = config.get("versions", {})
    installed = {
        "python": ".".join(platform.python_version().split(".")[:2]),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }
    problems = []
    for name, now in installed.items():
        was = saved.get(name)
        if name == "python" and was:
            was = ".".join(was.split(".")[:2])
        if was != now:
            problems.append(f"{name}: trained with {was}, installed {now}")
    return problems


# A pickled scikit-learn model is only guaranteed to work with the exact version that saved it.
# Loading without an error is not proof of compatibility, so a mismatch stops the app by default.
problems = version_problems()
if problems:
    message = "Training and deployment environments differ: " + "; ".join(problems)
    if os.environ.get("ALLOW_VERSION_MISMATCH") == "1":
        print("WARNING:", message)
    else:
        raise RuntimeError(message + ". Align requirements.txt and the Dockerfile with models/config.json "
                           "(the notebook does this when it saves the models), or set ALLOW_VERSION_MISMATCH=1 to override.")

with warnings.catch_warnings():
    warnings.simplefilter("error", InconsistentVersionWarning)   # never load a model silently across versions
    models = {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}

# One URL at a time: spreading 200 trees over worker threads costs more than it saves
if hasattr(models["Random Forest"], "set_params"):
    models["Random Forest"].set_params(n_jobs=1)

TUNED = "Tuned (low false-positive budget)"
DEFAULT = "Default (0.5)"

# What the person sees for a few of the observable characteristics (indicators only, not proof)
SHOWN = {
    "url_length": "URL length (characters)",
    "hostname_length": "Hostname length",
    "subdomain_count": "Number of subdomains",
    "digit_ratio": "Share of digits in URL",
    "has_ip_host": "Hostname is an IP address",
    "num_hyphens": "Hyphens",
    "num_at": "'@' symbols",
    "num_percent": "Percent-encoded characters",
    "path_depth": "Path depth",
    "query_length": "Query string length",
    "is_shortener": "Known link shortener",
}

EXAMPLES = [
    ["https://www.wikipedia.org/wiki/Phishing", "Random Forest", TUNED],
    ["https://github.com/anthropics", "Random Forest", TUNED],
    ["http://secure-login-account-verify123.example-bank-update.com/signin.php?user=482913&token=88213390211", "Random Forest", TUNED],
    ["http://192.168.45.12/paypal/login/confirm.html", "Logistic Regression", TUNED],
]


def triage(url, model_name, threshold_mode):
    clean, error = check_url_input(url)
    if error:
        return f"### ⚠️ {error}", None, None, ""

    start = time.perf_counter()
    frame = features_frame([clean])
    score = float(models[model_name].predict_proba(frame[FEATURES])[0, 1])
    elapsed_ms = (time.perf_counter() - start) * 1000

    key = "tuned" if threshold_mode == TUNED else "default"
    threshold = config["thresholds"][model_name][key]
    flagged = score >= threshold

    if flagged:
        verdict = ("### 🚩 Flagged for review\n"
                   "The URL text looks similar to phishing URLs seen in training. "
                   "Recommend an analyst checks it before anyone clicks.\n\n"
                   "_Heads up: this model tends to over-flag ordinary URLs with long paths (see Known limitation below)._")
    else:
        verdict = ("### ✅ Not flagged\n"
                   "The URL text does not look like known phishing patterns. "
                   "This is not a guarantee: phishing on legitimate domains or behind shorteners cannot be seen from the URL alone.")

    row = frame.iloc[0]
    details = pd.DataFrame(
        [{"Characteristic": label, "Value": round(float(row[k]), 3) if isinstance(row[k], float) else int(row[k])}
         for k, label in SHOWN.items()]
    )
    summary = pd.DataFrame([{
        "Phishing score": round(score, 4),
        "Threshold used": round(threshold, 4),
        "Model": model_name,
        "Inference time (ms)": round(elapsed_ms, 2),
    }])
    note = "These characteristics are indicators only, not proof of malicious intent. The URL was never visited."
    return verdict, summary, details, note


with gr.Blocks(title="Phishing URL Triage") as demo:
    gr.Markdown(
        "# Phishing URL Triage\n"
        "Paste a URL to get a phishing score based on the **text of the URL only**. "
        "The page is never visited. This is an advisory triage tool for a class project, not a full security product."
    )
    with gr.Row():
        with gr.Column():
            url_in = gr.Textbox(label="URL", placeholder="https://example.com/login", max_lines=1)
            model_in = gr.Dropdown(list(MODEL_FILES), value="Random Forest", label="Model")
            thr_in = gr.Radio([TUNED, DEFAULT], value=TUNED, label="Decision threshold")
            go = gr.Button("Analyze URL", variant="primary")
        with gr.Column():
            verdict_out = gr.Markdown()
            summary_out = gr.Dataframe(label="Result", interactive=False)
    details_out = gr.Dataframe(label="Observable URL characteristics", interactive=False)
    note_out = gr.Markdown()

    gr.Markdown(
        "**Known limitation:** the training data's legitimate URLs were mostly bare homepages, so the models partly learned "
        "that 'has a path' means phishing. On an independent dataset they flagged most legitimate URLs. "
        "Treat results as a triage hint, not a verdict, and expect false alarms on normal URLs with paths."
    )
    gr.Examples(EXAMPLES, inputs=[url_in, model_in, thr_in], label="Example URLs")
    go.click(triage, [url_in, model_in, thr_in], [verdict_out, summary_out, details_out, note_out])
    url_in.submit(triage, [url_in, model_in, thr_in], [verdict_out, summary_out, details_out, note_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
