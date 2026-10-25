"""
Automated tests for CipherMind Encrypted Vault & Threshold Secret Sharing.
All test values are strictly synthetic.
"""

import copy
import sys
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.privacy import redact_sensitive_data
from app.vault import (
    EncryptedVault,
    InsufficientSharesError,
    InvalidShareError,
    VaultAuthenticationError,
    VaultError,
    VaultManager,
    default_vault_manager,
    restore_text,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Cryptographic & Secret-Sharing Unit Tests
# ---------------------------------------------------------------------------

def test_successful_encryption_and_decryption_with_valid_shares():
    """Verify that a mapping is correctly encrypted and decrypted with 2 valid shares."""
    manager = VaultManager()
    synthetic_mapping = {
        "[EMAIL_1]": "synthetic.alice@example.com",
        "[PASSWORD_1]": "syntheticSecretPassword123!",
    }

    vault, shares = manager.create_vault(synthetic_mapping)

    assert len(shares) == 3
    assert vault.threshold == 2
    assert vault.total_shares == 3
    assert set(vault.placeholders) == {"[EMAIL_1]", "[PASSWORD_1]"}

    # Recover using shares 0 and 1
    recovered = manager.recover_mapping(vault, [shares[0], shares[1]])
    assert recovered == synthetic_mapping


def test_recovery_with_each_valid_two_share_combination():
    """Verify that ANY 2 of the 3 shares can independently reconstruct the master key."""
    manager = VaultManager()
    synthetic_mapping = {
        "[API_KEY_1]": "sk-proj-syntheticKey1234567890abcdef",
        "[PHONE_1]": "+1-555-123-4567",
    }

    vault, shares = manager.create_vault(synthetic_mapping)
    s0, s1, s2 = shares[0], shares[1], shares[2]

    # Combination 1: (0, 1)
    rec_01 = manager.recover_mapping(vault, [s0, s1])
    assert rec_01 == synthetic_mapping

    # Combination 2: (0, 2)
    rec_02 = manager.recover_mapping(vault, [s0, s2])
    assert rec_02 == synthetic_mapping

    # Combination 3: (1, 2)
    rec_12 = manager.recover_mapping(vault, [s1, s2])
    assert rec_12 == synthetic_mapping

    # All 3 shares should also succeed
    rec_all = manager.recover_mapping(vault, [s0, s1, s2])
    assert rec_all == synthetic_mapping


def test_failure_with_only_one_share():
    """Verify that 1 share is strictly insufficient to reconstruct the master key."""
    manager = VaultManager()
    synthetic_mapping = {"[EMAIL_1]": "synthetic.single@example.com"}

    vault, shares = manager.create_vault(synthetic_mapping)

    # Attempt recovery with only 1 share
    with pytest.raises(InsufficientSharesError) as exc_info:
        manager.recover_mapping(vault, [shares[0]])
    assert "Insufficient shares" in str(exc_info.value)

    # Empty list of shares
    with pytest.raises(InsufficientSharesError):
        manager.recover_mapping(vault, [])


def test_failure_with_duplicate_shares():
    """Verify that providing duplicate shares does not satisfy the threshold."""
    manager = VaultManager()
    synthetic_mapping = {"[PASSWORD_1]": "syntheticPass123"}

    vault, shares = manager.create_vault(synthetic_mapping)

    # Passing the exact same share twice
    with pytest.raises(InvalidShareError) as exc_info:
        manager.recover_mapping(vault, [shares[0], shares[0]])
    assert "Duplicate shares" in str(exc_info.value)


def test_invalid_and_malformed_shares():
    """Verify rejection of malformed, non-hex, or truncated shares."""
    manager = VaultManager()
    synthetic_mapping = {"[API_KEY_1]": "ghp_syntheticToken1234567890abcdef"}

    vault, shares = manager.create_vault(synthetic_mapping)

    # 1. Non-hex string of expected length
    with pytest.raises(InvalidShareError) as exc_info:
        manager.recover_mapping(vault, [shares[0], "Z" * 66])
    assert "invalid hexadecimal" in str(exc_info.value)

    # 2. Truncated hex string (wrong length)
    truncated = shares[1][:20]
    with pytest.raises(InvalidShareError) as exc_info:
        manager.recover_mapping(vault, [shares[0], truncated])
    assert "invalid length" in str(exc_info.value)

    # 3. Empty share string
    with pytest.raises(InvalidShareError):
        manager.recover_mapping(vault, [shares[0], "   "])


def test_modified_ciphertext_and_authentication_failure():
    """Verify that AES-256-GCM authenticated encryption rejects any tampering of ciphertext."""
    manager = VaultManager()
    synthetic_mapping = {"[EMAIL_1]": "synthetic.tamper@example.com"}

    vault, shares = manager.create_vault(synthetic_mapping)

    # Flip bits in the ciphertext
    ct_bytes = bytearray.fromhex(vault.ciphertext)
    ct_bytes[4] ^= 0xFF
    tampered_vault = EncryptedVault(
        vault_id=vault.vault_id,
        nonce=vault.nonce,
        ciphertext=ct_bytes.hex(),
        placeholders=vault.placeholders,
        created_at=vault.created_at,
        threshold=vault.threshold,
        total_shares=vault.total_shares,
    )

    with pytest.raises(VaultAuthenticationError) as exc_info:
        manager.recover_mapping(tampered_vault, [shares[0], shares[1]])
    assert "authentication failed" in str(exc_info.value).lower()


def test_corrupted_share_causes_authentication_failure():
    """Verify that a corrupted share produces an incorrect key that fails AES-GCM authentication."""
    manager = VaultManager()
    synthetic_mapping = {"[PASSWORD_1]": "syntheticTargetSecret123"}

    vault, shares = manager.create_vault(synthetic_mapping)

    # Corrupt share 1
    corrupted_share_bytes = bytearray.fromhex(shares[1])
    corrupted_share_bytes[15] ^= 0xAA
    corrupted_share_hex = corrupted_share_bytes.hex()

    with pytest.raises(VaultAuthenticationError):
        manager.recover_mapping(vault, [shares[0], corrupted_share_hex])


def test_unique_nonce_per_encryption_operation():
    """Verify that encrypting the same mapping repeatedly produces distinct nonces and ciphertexts."""
    manager = VaultManager()
    synthetic_mapping = {"[EMAIL_1]": "synthetic.repeat@example.com"}

    vault1, _ = manager.create_vault(synthetic_mapping)
    vault2, _ = manager.create_vault(synthetic_mapping)

    # Nonces must never be reused
    assert vault1.nonce != vault2.nonce
    # Ciphertexts must differ even for identical plaintext
    assert vault1.ciphertext != vault2.ciphertext


def test_empty_mapping_rejected():
    """Verify that creating a vault with an empty mapping raises ValueError."""
    manager = VaultManager()
    with pytest.raises(ValueError) as exc_info:
        manager.create_vault({})
    assert "empty mapping" in str(exc_info.value)


def test_invalid_threshold_parameters_rejected():
    """Verify parameter bounds enforcement for threshold and total_shares."""
    manager = VaultManager()
    mapping = {"[EMAIL_1]": "test@example.com"}

    # threshold < 2
    with pytest.raises(ValueError):
        manager.create_vault(mapping, threshold=1, total_shares=3)

    # threshold > total_shares
    with pytest.raises(ValueError):
        manager.create_vault(mapping, threshold=4, total_shares=3)


def test_integration_privacy_engine_and_vault():
    """Verify end-to-end integration: prompt inspection -> vault creation -> threshold recovery."""
    prompt = (
        "User synthetic.alice@example.com with password=syntheticPass987 "
        "and API key sk-proj-syntheticKey1234567890abcdef."
    )
    redaction = redact_sensitive_data(prompt)

    assert "[EMAIL_1]" in redaction.redacted_text
    assert "[PASSWORD_1]" in redaction.redacted_text
    assert "[API_KEY_1]" in redaction.redacted_text

    # Extract vault mapping from redaction
    assert len(redaction.vault_mapping) == 3

    # Encrypt into vault
    manager = VaultManager()
    vault, shares = manager.create_vault(redaction.vault_mapping)

    # Plaintext secrets must not appear in vault object
    vault_str = repr(vault)
    for raw_secret in redaction.vault_mapping.values():
        assert raw_secret not in vault_str

    # Recover mapping with threshold shares
    recovered_mapping = manager.recover_mapping(vault, [shares[0], shares[2]])
    assert recovered_mapping == redaction.vault_mapping


# ---------------------------------------------------------------------------
# API Endpoint Tests (/api/vault)
# ---------------------------------------------------------------------------

def test_api_get_vault_metadata_and_no_secret_leakage():
    """Verify GET /api/vault/{vault_id} returns metadata without exposing secrets, master key, or shares."""
    raw_secret = "syntheticSuperSecret999!"
    raw_email = "synthetic.meta@example.com"
    mapping = {
        "[EMAIL_1]": raw_email,
        "[PASSWORD_1]": raw_secret,
    }

    vault, shares = default_vault_manager.create_vault(mapping)

    response = client.get(f"/api/vault/{vault.vault_id}")
    assert response.status_code == 200
    data = response.json()

    assert data["vault_id"] == vault.vault_id
    assert data["status"] == "encrypted"
    assert data["threshold"] == 2
    assert data["total_shares"] == 3
    assert set(data["placeholders"]) == {"[EMAIL_1]", "[PASSWORD_1]"}

    # CRITICAL: Verify NO secrets, master keys, or shares in the API response
    response_text = response.text
    assert raw_secret not in response_text
    assert raw_email not in response_text
    for s in shares:
        assert s not in response_text


def test_api_get_nonexistent_vault_returns_404():
    """Verify 404 is returned for unknown vault ID."""
    response = client.get("/api/vault/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_api_verify_vault_shares_success():
    """Verify POST /api/vault/verify returns 200 without exposing secrets or keys."""
    raw_secret = "syntheticVerifyPass123"
    mapping = {"[PASSWORD_1]": raw_secret}

    vault, shares = default_vault_manager.create_vault(mapping)

    payload = {
        "vault_id": vault.vault_id,
        "shares": [shares[0], shares[1]],
    }
    response = client.post("/api/vault/verify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["vault_id"] == vault.vault_id
    assert data["status"] == "authenticated"
    assert data["threshold_met"] is True
    assert data["placeholder_count"] == 1

    # CRITICAL: Response must NOT leak secrets, master key, or shares
    assert raw_secret not in response.text
    for s in shares:
        assert s not in response.text


def test_api_verify_vault_shares_insufficient():
    """Verify POST /api/vault/verify returns 400 when only 1 share is provided."""
    mapping = {"[EMAIL_1]": "synthetic.insufficient@example.com"}
    vault, shares = default_vault_manager.create_vault(mapping)

    payload = {
        "vault_id": vault.vault_id,
        "shares": [shares[0]],
    }
    response = client.post("/api/vault/verify", json=payload)
    assert response.status_code == 400
    assert "Insufficient shares" in response.json()["detail"]


def test_api_verify_vault_shares_corrupted_returns_401():
    """Verify POST /api/vault/verify returns 401 when shares fail authentication."""
    mapping = {"[EMAIL_1]": "synthetic.corrupt@example.com"}
    vault, shares = default_vault_manager.create_vault(mapping)

    # Corrupt share
    bad_share = bytearray.fromhex(shares[1])
    bad_share[10] ^= 0xFF

    payload = {
        "vault_id": vault.vault_id,
        "shares": [shares[0], bad_share.hex()],
    }
    response = client.post("/api/vault/verify", json=payload)
    assert response.status_code == 401
    assert "authentication failed" in response.json()["detail"].lower()


def test_api_verify_nonexistent_vault_returns_404():
    """Verify POST /api/vault/verify returns 404 for unknown vault ID."""
    payload = {
        "vault_id": "00000000-0000-0000-0000-000000000000",
        "shares": ["a" * 66, "b" * 66],
    }
    response = client.post("/api/vault/verify", json=payload)
    assert response.status_code == 404


def test_restore_text_explicit_local_recovery():
    """Verify explicit, local text restoration using threshold shares."""
    manager = VaultManager()
    synthetic_mapping = {
        "[EMAIL_1]": "synthetic.restore@example.com",
        "[PASSWORD_1]": "syntheticP@ss999",
    }
    vault, shares = manager.create_vault(synthetic_mapping)
    redacted_text = "Contact [EMAIL_1] with credentials [PASSWORD_1] for access."

    # Successful restoration with shares (0, 1)
    restored = restore_text(redacted_text, vault, [shares[0], shares[1]], vault_manager=manager)
    assert restored == "Contact synthetic.restore@example.com with credentials syntheticP@ss999 for access."

    # Restoration fails with only 1 share
    with pytest.raises(InsufficientSharesError):
        restore_text(redacted_text, vault, [shares[0]], vault_manager=manager)

    # Restoration fails with corrupted share
    bad_share = bytearray.fromhex(shares[1])
    bad_share[5] ^= 0xFF
    with pytest.raises(VaultAuthenticationError):
        restore_text(redacted_text, vault, [shares[0], bad_share.hex()], vault_manager=manager)


def test_non_string_and_invalid_share_types_rejected():
    """Verify rejection of non-string and invalid share elements."""
    manager = VaultManager()
    mapping = {"[EMAIL_1]": "synthetic.types@example.com"}
    vault, shares = manager.create_vault(mapping)

    # Non-string element (int)
    with pytest.raises(InvalidShareError) as exc_info:
        manager.recover_mapping(vault, [shares[0], 12345])  # type: ignore
    assert "All shares must be non-empty strings" in str(exc_info.value)

    # Empty string element
    with pytest.raises(InvalidShareError) as exc_info:
        manager.recover_mapping(vault, [shares[0], ""])
    assert "All shares must be non-empty strings" in str(exc_info.value)


def test_api_verify_bounded_inputs_prevent_dos():
    """Verify that excessive share counts and oversized vault_ids are rejected (DoS protection)."""
    # 1. More than 10 shares in verify payload should be rejected with HTTP 422
    payload_too_many_shares = {
        "vault_id": "00000000-0000-0000-0000-000000000000",
        "shares": ["a" * 66] * 11,
    }
    res1 = client.post("/api/vault/verify", json=payload_too_many_shares)
    assert res1.status_code == 422

    # 2. Oversized vault_id (> 64 chars) should be rejected
    payload_oversized_vault = {
        "vault_id": "x" * 65,
        "shares": ["a" * 66, "b" * 66],
    }
    res2 = client.post("/api/vault/verify", json=payload_oversized_vault)
    assert res2.status_code == 422

    # 3. GET /api/vault/{vault_id} with oversized vault_id returns 404
    res3 = client.get(f"/api/vault/{'x' * 65}")
    assert res3.status_code == 404

