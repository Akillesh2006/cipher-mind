"""
Automated tests for CipherMind Privacy Engine and /api/privacy/inspect endpoint.
All test values are strictly synthetic.
"""

import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.privacy import MAX_TEXT_LENGTH, PrivacyEngine, redact_sensitive_data

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit Tests for PrivacyEngine Categories
# ---------------------------------------------------------------------------

def test_engine_email_detection():
    engine = PrivacyEngine()
    text = "Please reach out to synthetic.user@example.com or dev-team+ops@internal.corp.org."
    result = engine.redact(text)

    assert "synthetic.user@example.com" not in result.redacted_text
    assert "dev-team+ops@internal.corp.org" not in result.redacted_text
    assert "[EMAIL_1]" in result.redacted_text
    assert "[EMAIL_2]" in result.redacted_text
    assert result.total_findings == 2

    # Check findings metadata
    email_findings = [f for f in result.findings if f.category == "EMAIL"]
    assert len(email_findings) == 2
    assert email_findings[0].placeholder == "[EMAIL_1]"
    assert email_findings[0].count == 1
    assert email_findings[1].placeholder == "[EMAIL_2]"
    assert email_findings[1].count == 1


def test_engine_phone_detection_and_conservative_boundaries():
    engine = PrivacyEngine()
    # Includes standard US formatted, international, and E.164 numbers
    # Also includes dates, IP addresses, and standalone numbers that MUST NOT be matched
    text = (
        "Call (555) 123-4567 or +44 20 7946 0958 or +15559876543. "
        "Do not match date 2026-10-25 or IP 192.168.1.1 or math 10-20-30."
    )
    result = engine.redact(text)

    assert "(555) 123-4567" not in result.redacted_text
    assert "+44 20 7946 0958" not in result.redacted_text
    assert "+15559876543" not in result.redacted_text

    assert "[PHONE_1]" in result.redacted_text
    assert "[PHONE_2]" in result.redacted_text
    assert "[PHONE_3]" in result.redacted_text

    # Ensure non-phone patterns were preserved untouched
    assert "2026-10-25" in result.redacted_text
    assert "192.168.1.1" in result.redacted_text
    assert "10-20-30" in result.redacted_text
    assert result.total_findings == 3


def test_engine_api_keys_and_bearer_tokens():
    engine = PrivacyEngine()
    text = (
        "OpenAI key: sk-proj-syntheticOpenAiKey1234567890abcdef. "
        "GitHub token: ghp_syntheticGitHubToken1234567890abcdef. "
        "AWS key: AKIAIOSFODNN7EXAMPLE. "
        "Bearer token: Bearer syntheticBearerTokenValue123456789. "
        "Generic assign: api_key='syntheticGenericSecretKey12345'."
    )
    result = engine.redact(text)

    assert "sk-proj-syntheticOpenAiKey1234567890abcdef" not in result.redacted_text
    assert "ghp_syntheticGitHubToken1234567890abcdef" not in result.redacted_text
    assert "AKIAIOSFODNN7EXAMPLE" not in result.redacted_text
    assert "syntheticBearerTokenValue123456789" not in result.redacted_text
    assert "syntheticGenericSecretKey12345" not in result.redacted_text

    # Bearer keyword should be preserved, only token value redacted
    assert "Bearer [API_KEY_" in result.redacted_text

    api_key_findings = [f for f in result.findings if f.category == "API_KEY"]
    assert len(api_key_findings) == 5
    assert result.total_findings == 5


def test_engine_passwords_and_explicit_secrets():
    engine = PrivacyEngine()
    text = (
        "Login with password=syntheticPassword123! and "
        "db_pass: 'syntheticDbSecretPass' and "
        "client_secret=syntheticClientSecret987."
    )
    result = engine.redact(text)

    assert "syntheticPassword123!" not in result.redacted_text
    assert "syntheticDbSecretPass" not in result.redacted_text
    assert "syntheticClientSecret987" not in result.redacted_text

    assert "password=[PASSWORD_" in result.redacted_text
    assert "db_pass: '[PASSWORD_" in result.redacted_text

    password_findings = [f for f in result.findings if f.category == "PASSWORD"]
    assert len(password_findings) == 3


def test_engine_multiple_categories_in_one_prompt():
    engine = PrivacyEngine()
    text = (
        "Contact synthetic.alice@example.com at +1-555-123-4567. "
        "Use api_key=sk-proj-syntheticApiKey1234567890 and password=syntheticSecretPass99."
    )
    result = engine.redact(text)

    assert "[EMAIL_1]" in result.redacted_text
    assert "[PHONE_1]" in result.redacted_text
    assert "[API_KEY_1]" in result.redacted_text
    assert "[PASSWORD_1]" in result.redacted_text

    categories = {f.category for f in result.findings}
    assert categories == {"EMAIL", "PHONE", "API_KEY", "PASSWORD"}
    assert result.total_findings == 4


def test_engine_repeated_values_consistency():
    engine = PrivacyEngine()
    text = (
        "First notice: synthetic.admin@example.com is our contact. "
        "Second notice: send confirmation to synthetic.admin@example.com. "
        "Also password=syntheticPassword777 was given; remember that syntheticPassword777 expires tomorrow."
    )
    result = engine.redact(text)

    # synthetic.admin@example.com must get the SAME placeholder [EMAIL_1] in both spots
    assert result.redacted_text.count("[EMAIL_1]") == 2
    assert "[EMAIL_2]" not in result.redacted_text

    # syntheticPassword777 must get the SAME placeholder [PASSWORD_1] in both spots
    assert result.redacted_text.count("[PASSWORD_1]") == 2

    # Verify counts in findings
    email_f = next(f for f in result.findings if f.placeholder == "[EMAIL_1]")
    assert email_f.count == 2

    pass_f = next(f for f in result.findings if f.placeholder == "[PASSWORD_1]")
    assert pass_f.count == 2

    assert result.total_findings == 4


def test_engine_overlapping_patterns_resolution():
    engine = PrivacyEngine()
    # 1. Generic API key assignment overlapping with OpenAI provider pattern
    text1 = "api_key=sk-proj-syntheticKey1234567890abcdef12"
    result1 = engine.redact(text1)
    assert result1.redacted_text == "api_key=[API_KEY_1]"
    assert result1.total_findings == 1

    # 2. Bearer token overlapping with JWT pattern
    jwt_sample = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.syntheticSignature99999"
    text2 = f"Authorization: Bearer {jwt_sample}"
    result2 = engine.redact(text2)
    assert result2.redacted_text == "Authorization: Bearer [API_KEY_1]"
    assert result2.total_findings == 1

    # 3. Password assignment overlapping with an email
    text3 = "password=synthetic.user@example.com"
    result3 = engine.redact(text3)
    assert result3.redacted_text == "password=[PASSWORD_1]"
    assert result3.total_findings == 1


def test_engine_no_sensitive_matches():
    engine = PrivacyEngine()
    text = "The quick brown fox jumps over the lazy dog in October 2026."
    result = engine.redact(text)

    assert result.redacted_text == text
    assert result.findings == []
    assert result.total_findings == 0


def test_engine_empty_input():
    engine = PrivacyEngine()
    result = engine.redact("")
    assert result.redacted_text == ""
    assert result.findings == []
    assert result.total_findings == 0


def test_engine_oversized_input_raises_value_error():
    engine = PrivacyEngine()
    oversized = "A" * (MAX_TEXT_LENGTH + 1)
    try:
        engine.redact(oversized)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "exceeds maximum allowed limit" in str(e)


# ---------------------------------------------------------------------------
# API Endpoint Tests (/api/privacy/inspect)
# ---------------------------------------------------------------------------

def test_api_inspect_success_status_and_response_structure():
    payload = {
        "text": "Hello, contact synthetic.alice@example.com or call (555) 123-4567."
    }
    response = client.post("/api/privacy/inspect", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "redacted_text" in data
    assert "findings" in data
    assert "total_findings" in data

    assert "[EMAIL_1]" in data["redacted_text"]
    assert "[PHONE_1]" in data["redacted_text"]
    assert data["total_findings"] == 2

    # Check structure of each finding item
    for item in data["findings"]:
        assert "category" in item
        assert "placeholder" in item
        assert "count" in item


def test_api_inspect_never_leaks_secrets_in_metadata():
    raw_secret = "syntheticSecretPassword987"
    raw_email = "synthetic.sensitive@example.com"
    raw_key = "sk-proj-syntheticKey1234567890abcdef12"

    payload = {
        "text": f"User: {raw_email}, Key: {raw_key}, Pass: password={raw_secret}"
    }
    response = client.post("/api/privacy/inspect", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Raw secrets must NOT be in redacted text
    assert raw_secret not in data["redacted_text"]
    assert raw_email not in data["redacted_text"]
    assert raw_key not in data["redacted_text"]

    # Raw secrets must NOT be anywhere in findings metadata
    findings_str = str(data["findings"])
    assert raw_secret not in findings_str
    assert raw_email not in findings_str
    assert raw_key not in findings_str


def test_api_inspect_empty_text():
    response = client.post("/api/privacy/inspect", json={"text": ""})
    assert response.status_code == 200
    data = response.json()

    assert data["redacted_text"] == ""
    assert data["findings"] == []
    assert data["total_findings"] == 0


def test_api_inspect_oversized_payload():
    oversized_text = "x" * (MAX_TEXT_LENGTH + 1)
    response = client.post("/api/privacy/inspect", json={"text": oversized_text})
    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"]


def test_api_inspect_invalid_request_bodies():
    # 1. Missing required 'text' field
    res1 = client.post("/api/privacy/inspect", json={"wrong_field": "hello"})
    assert res1.status_code == 422

    # 2. 'text' field is wrong data type (integer instead of string)
    res2 = client.post("/api/privacy/inspect", json={"text": 12345})
    assert res2.status_code == 422

    # 3. Empty JSON object
    res3 = client.post("/api/privacy/inspect", json={})
    assert res3.status_code == 422

    # 4. Non-JSON body
    res4 = client.post(
        "/api/privacy/inspect",
        content="plain text body",
        headers={"Content-Type": "application/json"},
    )
    assert res4.status_code == 422


def test_health_endpoints_remain_unaffected():
    """Verify no regressions in existing health endpoints."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "running"

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
