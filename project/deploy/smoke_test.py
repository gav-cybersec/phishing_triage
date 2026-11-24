"""
End-to-end check of the deployment package in the environment it will run in.
Run:  python smoke_test.py      (the Docker build runs this too, so a bad package fails the build)

It checks that the library versions match training, both saved models load and score, valid URLs
get a verdict, and bad input is rejected with a message instead of crashing.
"""
import app

failures = []

print("Environment matches training:", not app.version_problems())

for name, model in app.models.items():
    for url in ["https://www.wikipedia.org", "http://192.168.45.12/paypal/login/confirm.html",
                "http://secure-login-account-verify123.example-bank-update.com/signin.php?user=482913"]:
        for mode in (app.TUNED, app.DEFAULT):
            verdict, summary, details, note = app.triage(url, name, mode)
            if not (verdict.startswith("### 🚩") or verdict.startswith("### ✅")):
                failures.append(f"{name} / {url}: unexpected verdict {verdict[:40]!r}")
                continue
            score = float(summary.iloc[0]["Phishing score"])
            if not 0.0 <= score <= 1.0:
                failures.append(f"{name} / {url}: score out of range ({score})")
            if len(details) == 0:
                failures.append(f"{name} / {url}: no URL characteristics returned")

for bad in ["", "hello", "javascript:alert(1)", "https://example.com:abc/a", "https://999.999.999.999/a"]:
    verdict, summary, details, note = app.triage(bad, "Random Forest", app.TUNED)
    if not verdict.startswith("### ⚠️") or summary is not None:
        failures.append(f"bad input {bad!r} was not rejected cleanly")

if failures:
    print(f"{len(failures)} smoke test failure(s):")
    for f in failures:
        print("  -", f)
    raise SystemExit(1)
print("Smoke test passed.")
