"""
Plain-Python tests for features.py (no pytest needed).  Run:  python test_features.py
Covers URL normalization, input validation (app input), and feature extraction.
"""
import math

from features import (
    FEATURES, normalize_url, check_url_input, extract_features, features_frame, registered_domain,
)

failures = []


def check(condition, message):
    if not condition:
        failures.append(message)


# ---------------------------------------------------------------- normalization (a modeling view, not URL equivalence)
check(normalize_url("HTTP://www.Example.com/") == "example.com", "scheme, www and root slash removed, host lowercased")
check(normalize_url("https://example.com") == "example.com", "https removed")
check(normalize_url("example.com/a/") == "example.com/a/", "non-root trailing slash kept")
check(normalize_url("example.com/?x=1") == "example.com/?x=1", "query kept")
check(normalize_url("HTTP://www.Example.com/Account?Token=AbC") == "example.com/Account?Token=AbC", "path and query keep their case")
check(normalize_url("example.com/Account?Token=AbC") != normalize_url("example.com/account?token=abc"), "different-case paths are different keys")
check(normalize_url("https://User:PassWord@WWW.Example.com:8080/x") == "User:PassWord@example.com:8080/x", "user info keeps case, host lowercased")
check(normalize_url("http://www.example.com:8080/") == "example.com:8080", "root slash after port removed")

# ---------------------------------------------------------------- input validation
VALID = [
    "https://example.com",
    "http://www.Example.com/a?b=1",
    "example.com/login",
    "https://sub.example.co.uk/a/b",
    "https://example.com:8080/a",
    "example.com:8080/a",
    "192.168.1.1/admin",
    "http://[2001:db8::1]/x",
    "http://[2001:db8::1]:8080/x",
    "http://user:pw@example.com/",
    "https://münchen.de/x",
    "bit.ly/abc123",
    "http://2130706433/a",
    "https://example.com/" + "a" * 100,
]
INVALID = {
    "empty": "",
    "blank": "   ",
    "none": None,
    "single word": "hello",
    "localhost": "localhost",
    "localhost with port": "localhost:8080",
    "javascript scheme": "javascript:alert(1)",
    "mailto scheme": "mailto:a@b.com",
    "ftp scheme": "ftp://example.com/file",
    "file scheme": "file:///etc/passwd",
    "non-numeric port": "https://example.com:abc/a",
    "port zero": "https://example.com:0/",
    "port too large": "https://example.com:70000/",
    "bad ipv4": "https://999.999.999.999/a",
    "short ipv4": "http://1.2.3/a",
    "bad ipv6": "http://[zzz]/x",
    "unclosed ipv6": "http://[::1",
    "space": "http://exa mple.com",
    "newline": "http://example.com/a\nb",
    "control character": "http://example.com/\x00",
    "no host": "https://",
    "user info only": "http://user@",
    "empty label": "http://a..com",
    "leading dot": "http://.com",
    "leading hyphen": "http://-bad.com",
    "trailing hyphen": "http://bad-.com",
    "numeric tld": "http://example.123",
    "long label": "http://" + "a" * 64 + ".com",
    "long host": "http://" + ".".join(["a" * 60] * 5) + ".com",
    "too long": "https://example.com/" + "a" * 2100,
}

for url in VALID:
    clean, error = check_url_input(url)
    check(error is None and clean == url, f"should be valid: {url!r} (got {error})")

for name, url in INVALID.items():
    clean, error = check_url_input(url)
    check(clean is None and bool(error), f"should be rejected ({name}): {url!r}")

# ---------------------------------------------------------------- feature extraction
for url in VALID:
    values = extract_features(url)
    check(list(values) == FEATURES, f"feature names/order wrong for {url!r}")
    check(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()), f"non-finite feature for {url!r}")

check(extract_features("http://192.168.1.1/a")["has_ip_host"] == 1, "IPv4 host detected")
check(extract_features("https://example.com")["has_ip_host"] == 0, "domain host is not an IP")
check(extract_features("http://x.com/")["num_slashes"] == extract_features("x.com")["num_slashes"] == 0, "root slash does not change num_slashes")
check(extract_features("example.com/a")["path_depth"] == 1, "path depth")
check(extract_features("a.b.example.com")["subdomain_count"] == 2, "subdomain count")
check(extract_features("a.example.co.uk")["subdomain_count"] == 1, "two-part suffix handled")
check(registered_domain("a.b.example.co.uk") == "example.co.uk", "registered domain for co.uk")

frame = features_frame(["https://example.com/a", "", "https://"])
check(list(frame["ok"]) == [True, False, False], "features_frame marks unparseable rows")

if failures:
    print(f"{len(failures)} test(s) FAILED:")
    for f in failures:
        print("  -", f)
    raise SystemExit(1)
print(f"All tests passed ({len(VALID)} valid, {len(INVALID)} invalid inputs, plus normalization and feature checks).")
