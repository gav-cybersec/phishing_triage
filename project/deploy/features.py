"""
URL-only feature extraction.

This one file is imported by BOTH the training notebook and the Gradio app,
so training and deployment always compute features the exact same way.
Only the URL text is used. The page is never visited and redirects are never followed.
"""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

MAX_URL_LENGTH = 2048

FEATURES = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "subdomain_count",
    "tld_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at",
    "num_question",
    "num_equals",
    "num_ampersand",
    "num_percent",
    "path_depth",
    "digit_ratio",
    "letter_ratio",
    "special_char_ratio",
    "hostname_digit_ratio",
    "has_ip_host",
    "has_port",
    "is_shortener",
    "url_entropy",
]

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "tiny.cc", "shorturl.at",
}

# Common two-part public suffixes (a small list, not the full Public Suffix List)
TWO_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au",
    "co.jp", "co.in", "co.nz", "com.br", "com.cn", "com.mx", "co.za",
    "com.tr", "com.sg", "com.hk", "com.ar", "co.kr", "com.ua",
}

IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
ODD_IP = re.compile(r"^(0x[0-9a-f]+|\d{8,10})$")


SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")


def normalize_url(url):
    """
    The "model view" of a URL, used for features, de-duplication and overlap checks.

    These are MODELING decisions made so two differently collected datasets share one format.
    They are not claims that the resulting URLs are always equivalent:
      - the scheme (http/https) is removed
      - a leading www. is removed from the hostname
      - the hostname is lowercased (hostnames are case-insensitive)
      - a lone trailing slash right after the hostname is removed (example.com/ -> example.com)
    The path, query and any user info keep their original case, because they can be case-sensitive.
    """
    url = SCHEME.sub("", str(url).strip()).lstrip("/")
    authority, rest = re.match(r"^([^/?#]*)(.*)$", url, flags=re.S).groups()
    userinfo, at, hostport = authority.rpartition("@")
    hostport = re.sub(r"^www\d{0,3}\.", "", hostport.lower())
    if rest == "/":
        rest = ""
    return f"{userinfo}{at}{hostport}{rest}"


def split_url(url):
    """Return (normalized url, parsed parts, hostname). Raises ValueError for unusable input."""
    clean = normalize_url(url)
    if not clean:
        raise ValueError("Empty URL")
    parts = urlsplit("//" + clean)
    host = parts.hostname or ""
    if not host:
        raise ValueError("No hostname found")
    return clean, parts, host


def is_ip_host(host):
    return bool(IPV4.match(host) or ODD_IP.match(host) or ":" in host)


def registered_domain(host):
    """Approximate registered domain (example.co.uk, example.com). Used for grouping and subdomain counts."""
    if is_ip_host(host):
        return host
    labels = host.split(".")
    if len(labels) < 2:
        return host
    if ".".join(labels[-2:]) in TWO_PART_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def entropy(text):
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_features(url):
    """Return a dict of FEATURES for one URL. Raises ValueError if the URL cannot be parsed."""
    clean, parts, host = split_url(url)
    n = len(clean)
    ip = is_ip_host(host)
    labels = host.split(".")

    try:
        has_port = int(parts.port is not None)
    except ValueError:
        has_port = 1

    digits = sum(ch.isdigit() for ch in clean)
    letters = sum(ch.isalpha() for ch in clean)
    host_digits = sum(ch.isdigit() for ch in host)
    reg = registered_domain(host)

    return {
        "url_length": n,
        "hostname_length": len(host),
        "path_length": len(parts.path),
        "query_length": len(parts.query),
        "subdomain_count": 0 if ip else max(len(labels) - len(reg.split(".")), 0),
        "tld_length": 0 if ip else len(labels[-1]),
        "num_dots": clean.count("."),
        "num_hyphens": clean.count("-"),
        "num_underscores": clean.count("_"),
        "num_slashes": clean.count("/"),
        "num_at": clean.count("@"),
        "num_question": clean.count("?"),
        "num_equals": clean.count("="),
        "num_ampersand": clean.count("&"),
        "num_percent": clean.count("%"),
        "path_depth": len([p for p in parts.path.split("/") if p]),
        "digit_ratio": digits / n,
        "letter_ratio": letters / n,
        "special_char_ratio": (n - digits - letters) / n,
        "hostname_digit_ratio": host_digits / len(host),
        "has_ip_host": int(ip),
        "has_port": has_port,
        "is_shortener": int(reg in SHORTENERS),
        "url_entropy": entropy(clean),
    }


def features_frame(urls):
    """
    Build a feature table for many URLs.
    Returns a DataFrame with the FEATURES columns plus an 'ok' column.
    Rows that could not be parsed have ok=False and NaN features.
    """
    rows, ok = [], []
    for url in urls:
        try:
            rows.append(extract_features(url))
            ok.append(True)
        except (ValueError, TypeError):
            rows.append({name: np.nan for name in FEATURES})
            ok.append(False)
    frame = pd.DataFrame(rows, columns=FEATURES)
    frame["ok"] = ok
    return frame


SUPPORTED_SCHEMES = {"http", "https"}
HOST_LABEL = re.compile(r"^(?!-)[a-z0-9_-]{1,63}(?<!-)$")


def _check_host(host, netloc):
    """Return an error message for an invalid hostname, or None if it is acceptable."""
    # IPv6 literal, written in square brackets
    if "[" in netloc or ":" in host:
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            return "Invalid IPv6 address."
        return None

    # Numeric / hex single-number IP forms (for example 2130706433) are real obfuscation tricks, so they are accepted
    if ODD_IP.match(host):
        return None

    # Dotted numbers must be a real IPv4 address (so 999.999.999.999 is rejected)
    if re.fullmatch(r"[0-9.]+", host):
        try:
            ipaddress.IPv4Address(host)
        except ValueError:
            return "Invalid IPv4 address."
        return None

    try:
        ascii_host = host.rstrip(".").encode("idna").decode("ascii")
    except UnicodeError:
        return "Hostname contains characters that cannot be used in a domain name."
    if len(ascii_host) > 253:
        return "Hostname is too long."

    labels = ascii_host.split(".")
    if len(labels) < 2:
        # Decision: single-label names (hello, localhost, intranet) are not supported by this tool
        return "Please enter a full domain name such as example.com (single-word hostnames are not supported)."
    if any(not HOST_LABEL.match(label) for label in labels):
        return "Hostname has an empty or invalid part (letters, digits, hyphens and underscores only, 63 characters per part, no leading or trailing hyphen)."
    if labels[-1].isdigit():
        return "The last part of a domain name cannot be all digits."
    return None


def check_url_input(text):
    """
    Validate user input for the app. Returns (clean_text, error_message).

    Accepts http/https URLs, with or without the scheme (example.com/login), IPv4 and bracketed IPv6 hosts,
    and optional numeric ports. Rejects other schemes, single-word hostnames, invalid IPs and invalid ports.
    The training pipeline stays lenient on purpose because dataset URLs are messy; only user input is strict.
    """
    if text is None or not str(text).strip():
        return None, "Please enter a URL."
    text = str(text).strip()
    if len(text) > MAX_URL_LENGTH:
        return None, f"URL is longer than {MAX_URL_LENGTH} characters."
    if re.search(r"[\s\x00-\x1f\x7f]", text):
        return None, "URLs cannot contain spaces, line breaks or control characters."

    explicit = re.match(r"^([A-Za-z][A-Za-z0-9+.\-]*)://", text)
    if explicit and explicit.group(1).lower() not in SUPPORTED_SCHEMES:
        return None, "Only http and https URLs are supported."
    if not explicit:
        # "scheme:something" without // (javascript:alert(1), mailto:a@b.com). A host:port such as example.com:8080 is fine.
        scheme_like = re.match(r"^([A-Za-z][A-Za-z0-9+.\-]*):", text)
        host_port = re.match(r"^[^/?#:@]+:\d+([/?#]|$)", text)
        if scheme_like and "." not in scheme_like.group(1) and not host_port:
            return None, "Only http and https URLs are supported."

    try:
        clean, parts, host = split_url(text)
    except ValueError as e:
        return None, f"Could not parse URL: {e}."

    try:
        port = parts.port
    except ValueError:
        return None, "The port must be a number between 1 and 65535."
    if port is not None and not 1 <= port <= 65535:
        return None, "The port must be a number between 1 and 65535."

    problem = _check_host(host, parts.netloc)
    if problem:
        return None, problem
    return text, None


def phish_score(model, urls):
    """Phishing probability for each URL. Models are trained with 1 = phishing."""
    frame = features_frame(urls)
    if not frame["ok"].all():
        raise ValueError("Some URLs could not be parsed")
    return model.predict_proba(frame[FEATURES])[:, 1]
