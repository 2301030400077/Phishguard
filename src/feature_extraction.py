"""Deterministic URL-only feature extraction. This module never makes network requests."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, urlsplit

FEATURE_NAMES = [
    "url_length", "hostname_length", "dot_count", "hyphen_count", "digit_count",
    "special_char_count", "subdomain_count", "has_ip_address", "has_at_symbol",
    "uses_https", "suspicious_keyword_count", "path_length", "query_length",
    "parameter_count",
]
SUSPICIOUS_KEYWORDS = (
    "login", "signin", "verify", "account", "secure", "update", "password",
    "bank", "confirm", "billing", "webscr", "auth", "recover",
)


def extract_features(url: str) -> dict[str, int]:
    """Return the stable numeric feature schema used by both training and inference."""
    if not isinstance(url, str):
        raise TypeError("URL must be text")
    value = url.strip()
    candidate = value if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value) else "//" + value
    try:
        parts = urlsplit(candidate)
        hostname = (parts.hostname or "").lower().rstrip(".")
    except (ValueError, UnicodeError):
        parts = urlsplit("")
        hostname = ""
    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0
    labels = [label for label in hostname.split(".") if label]
    try:
        parameter_count = len(parse_qsl(parts.query, keep_blank_values=True, max_num_fields=256))
    except (ValueError, TypeError):
        parameter_count = min(parts.query.count("&") + (1 if parts.query else 0), 256)
    lowered = value.lower()
    return {
        "url_length": len(value),
        "hostname_length": len(hostname),
        "dot_count": value.count("."),
        "hyphen_count": value.count("-"),
        "digit_count": sum(char.isdigit() for char in value),
        "special_char_count": sum(not char.isalnum() for char in value),
        "subdomain_count": max(0, len(labels) - 2) if not is_ip else 0,
        "has_ip_address": is_ip,
        "has_at_symbol": int("@" in value),
        "uses_https": int(parts.scheme.lower() == "https"),
        "suspicious_keyword_count": sum(keyword in lowered for keyword in SUSPICIOUS_KEYWORDS),
        "path_length": len(parts.path),
        "query_length": len(parts.query),
        "parameter_count": parameter_count,
    }


def feature_vector(url: str) -> list[int]:
    extracted = extract_features(url)
    return [extracted[name] for name in FEATURE_NAMES]
