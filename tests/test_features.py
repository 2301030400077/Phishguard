import pytest

from src.feature_extraction import FEATURE_NAMES, extract_features, feature_vector


def test_feature_schema_and_counts():
    url = "https://shop.example.com/account?user=a&next=b"
    result = extract_features(url)
    assert list(result) == FEATURE_NAMES
    assert result["uses_https"] == 1
    assert result["hostname_length"] == len("shop.example.com")
    assert result["subdomain_count"] == 1
    assert result["parameter_count"] == 2
    assert result["suspicious_keyword_count"] == 1
    assert feature_vector(url) == [result[name] for name in FEATURE_NAMES]


def test_ip_address_and_http_are_features_not_verdicts():
    result = extract_features("http://192.168.0.1/login")
    assert result["has_ip_address"] == 1
    assert result["uses_https"] == 0
    assert len(result) == len(FEATURE_NAMES)


def test_malformed_url_is_handled_without_network_access():
    result = extract_features("http://[broken")
    assert set(result) == set(FEATURE_NAMES)
    assert result["hostname_length"] == 0


def test_non_text_input_is_rejected():
    with pytest.raises(TypeError):
        extract_features(None)
