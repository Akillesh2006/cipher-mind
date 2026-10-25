"""
CipherMind Encrypted Vault & Threshold Secret Sharing
=====================================================

Provides authenticated encryption for sensitive placeholder mappings using
AES-256-GCM and threshold master-key recovery via Shamir's Secret Sharing (2-of-3).

Cryptographic Architecture
--------------------------
1. Master Key:
   - 256-bit (32-byte) cryptographically secure random key generated via secrets.token_bytes(32).
   - Zeroized and discarded from memory immediately after encryption and share splitting.
   - Never written to persistent storage or exposed in API responses.

2. Authenticated Encryption:
   - AES-256-GCM via the cryptography library (cryptography.hazmat.primitives.ciphers.aead.AESGCM).
   - Nonce: 96-bit (12-byte) cryptographically random nonce generated fresh for every operation (os.urandom(12)).
   - Authenticated Associated Data (AAD): Bound to vault_id to prevent ciphertext substitution.
   - Authentication Tag: 128-bit tag verified on decryption; tampered ciphertext is rejected with VaultAuthenticationError.

3. Threshold Secret Sharing:
   - Shamir's Secret Sharing over GF(256) via pyshamir (HashiCorp Vault port).
   - Total Shares: 3.
   - Recovery Threshold: 2.
   - Any 2 of 3 valid shares reconstruct the master key.
   - 1 share reveals mathematically zero information about the 256-bit master key.

Security Boundaries & Prototype Scope
-------------------------------------
- Access Control: Vault IDs (UUIDv4) are resource identifiers, NOT access control
  or authorization tokens. In the absence of an authentication/authorization layer,
  knowledge of a vault ID allows querying non-sensitive metadata (e.g. placeholder
  tags). Decryption of sensitive secrets remains cryptographically guarded by the
  threshold shares.
- Custody Model: Threshold secret sharing provides resilience against compromised individual
  custodians. However, if a single attacker controls all shares or compromises the host memory,
  threshold sharing provides no defensive benefit.
- Implementation Limitations: Software-based Shamir secret sharing over GF(256) in Python
  is not constant-time and is vulnerable to host-level micro-architectural timing side channels.
- Independent Audit: This prototype uses standard primitives (AES-GCM, pyshamir), but has
  NOT undergone formal third-party cryptographic auditing or security certification.
- Plaintext Secrecy: Original secret values are never included in logs, diagnostic metadata,
  or API responses.
"""

from dataclasses import dataclass
import json
import os
import secrets
import threading
import time
from typing import Dict, List, Optional, Set, Tuple
import uuid

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pyshamir import combine, split


class VaultError(Exception):
    """Base exception for all vault and secret-sharing errors."""
    pass


class InsufficientSharesError(VaultError):
    """Raised when fewer shares than the required threshold are provided."""
    pass


class InvalidShareError(VaultError):
    """Raised when shares are malformed, duplicate, or mathematically invalid."""
    pass


class VaultAuthenticationError(VaultError):
    """Raised when ciphertext authentication or share reconstruction fails."""
    pass


class VaultNotFoundError(VaultError):
    """Raised when a specified vault ID does not exist in memory."""
    pass


@dataclass(frozen=True)
class EncryptedVault:
    """Represents an authenticated, encrypted mapping vault.

    Attributes:
        vault_id: Unique identifier for the vault.
        nonce: Hex-encoded 12-byte initialization vector.
        ciphertext: Hex-encoded AES-256-GCM ciphertext including the 16-byte authentication tag.
        placeholders: List of placeholder names contained in the vault (e.g., ['[EMAIL_1]']).
        created_at: POSIX creation timestamp.
        threshold: Minimum shares required for recovery (default: 2).
        total_shares: Total shares generated (default: 3).
    """
    vault_id: str
    nonce: str
    ciphertext: str
    placeholders: List[str]
    created_at: float
    threshold: int = 2
    total_shares: int = 3


class VaultStore:
    """Thread-safe in-memory store for EncryptedVault instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vaults: Dict[str, EncryptedVault] = {}

    def put(self, vault: EncryptedVault) -> None:
        with self._lock:
            self._vaults[vault.vault_id] = vault

    def get(self, vault_id: str) -> Optional[EncryptedVault]:
        with self._lock:
            return self._vaults.get(vault_id)

    def delete(self, vault_id: str) -> bool:
        with self._lock:
            return self._vaults.pop(vault_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._vaults.clear()

    def count(self) -> int:
        with self._lock:
            return len(self._vaults)


class VaultManager:
    """Cryptographic manager for vault encryption, splitting, and threshold recovery."""

    def __init__(self, store: Optional[VaultStore] = None) -> None:
        self.store = store or VaultStore()

    def create_vault(
        self,
        mapping: Dict[str, str],
        threshold: int = 2,
        total_shares: int = 3,
    ) -> Tuple[EncryptedVault, List[str]]:
        """Encrypt a mapping of {placeholder: secret} and split the master key into Shamir shares.

        Args:
            mapping: Non-empty dictionary mapping placeholders to sensitive values.
            threshold: Minimum number of shares required to reconstruct master key (default 2).
            total_shares: Total number of shares to produce (default 3).

        Returns:
            Tuple containing:
                - EncryptedVault metadata and ciphertext (safe to store/transport).
                - List of hex-encoded secret shares to be distributed to independent custodians.

        Raises:
            ValueError: If mapping is empty or threshold parameters are invalid.
        """
        if not mapping:
            raise ValueError("Cannot create an encrypted vault with an empty mapping.")

        if threshold < 2:
            raise ValueError(f"Recovery threshold must be at least 2 (got {threshold}).")

        if threshold > total_shares:
            raise ValueError(
                f"Threshold ({threshold}) cannot exceed total shares ({total_shares})."
            )

        vault_id = str(uuid.uuid4())

        # 1. Generate cryptographically secure random 256-bit master key
        raw_key = bytearray(secrets.token_bytes(32))

        try:
            # 2. Generate unique 96-bit nonce for this encryption operation
            nonce_bytes = os.urandom(12)

            # 3. Serialize and encrypt mapping with AES-256-GCM
            plaintext_bytes = json.dumps(mapping, separators=(",", ":")).encode("utf-8")
            aad = vault_id.encode("utf-8")
            aesgcm = AESGCM(bytes(raw_key))
            ciphertext_bytes = aesgcm.encrypt(nonce_bytes, plaintext_bytes, aad)

            # 4. Split master key into Shamir shares
            raw_shares = split(bytes(raw_key), total_shares, threshold)
            hex_shares = [s.hex() for s in raw_shares]

        finally:
            # 5. Overwrite and erase plaintext master key from memory immediately
            for i in range(len(raw_key)):
                raw_key[i] = 0
            del raw_key

        vault = EncryptedVault(
            vault_id=vault_id,
            nonce=nonce_bytes.hex(),
            ciphertext=ciphertext_bytes.hex(),
            placeholders=sorted(mapping.keys()),
            created_at=time.time(),
            threshold=threshold,
            total_shares=total_shares,
        )

        self.store.put(vault)
        return vault, hex_shares

    def recover_mapping(
        self,
        vault: EncryptedVault,
        shares: List[str],
    ) -> Dict[str, str]:
        """Reconstruct the master key using threshold shares and decrypt the vault mapping.

        Args:
            vault: The EncryptedVault instance to decrypt.
            shares: List of hex-encoded secret shares.

        Returns:
            Decrypted dictionary mapping placeholders to their original values.

        Raises:
            InsufficientSharesError: If fewer than threshold shares are provided.
            InvalidShareError: If shares are duplicate, malformed, or invalid hex.
            VaultAuthenticationError: If shares reconstruct an invalid key or ciphertext was modified.
        """
        if not shares or len(shares) < vault.threshold:
            received = len(shares) if shares else 0
            raise InsufficientSharesError(
                f"Insufficient shares provided: received {received}, required threshold is {vault.threshold}."
            )

        # Validate that all items in shares are non-empty strings
        if not all(isinstance(s, str) and s.strip() for s in shares):
            raise InvalidShareError("All shares must be non-empty strings.")

        # Validate hex length bounds: 33-byte share must be exactly 66 hex characters
        for idx, s in enumerate(shares):
            clean_s = s.strip()
            if len(clean_s) != 66:
                raise InvalidShareError(
                    f"Share at index {idx} has invalid length ({len(clean_s)} hex chars, expected 66)."
                )

        # Deduplicate and verify threshold of unique shares
        stripped_shares = [s.strip() for s in shares]
        unique_shares = list(dict.fromkeys(stripped_shares))
        if len(unique_shares) < len(stripped_shares):
            if len(unique_shares) < vault.threshold:
                raise InvalidShareError(
                    f"Duplicate shares provided. Found {len(unique_shares)} unique share(s), "
                    f"but threshold requires {vault.threshold} distinct shares."
                )

        # Decode hex shares
        raw_shares: List[bytearray] = []
        for idx, s in enumerate(unique_shares):
            try:
                b = bytearray.fromhex(s)
            except ValueError:
                raise InvalidShareError(f"Share at index {idx} contains invalid hexadecimal encoding.")
            raw_shares.append(b)

        # Combine threshold shares to reconstruct master key
        try:
            recovered_key = combine(raw_shares[:vault.threshold])
        except ValueError as e:
            raise InvalidShareError(f"Secret-sharing combination error: {str(e)}")

        try:
            # Decrypt and authenticate using AES-256-GCM
            nonce_bytes = bytes.fromhex(vault.nonce)
            ciphertext_bytes = bytes.fromhex(vault.ciphertext)
            aad = vault.vault_id.encode("utf-8")

            aesgcm = AESGCM(bytes(recovered_key))
            decrypted_data = aesgcm.decrypt(nonce_bytes, ciphertext_bytes, aad)
            decrypted_mapping: Dict[str, str] = json.loads(decrypted_data.decode("utf-8"))
            return decrypted_mapping

        except InvalidTag:
            raise VaultAuthenticationError(
                "Vault authentication failed: cryptographic tag mismatch. "
                "The ciphertext has been modified or an invalid share corrupted key reconstruction."
            )
        finally:
            # Clean up recovered key and raw share buffers in memory immediately
            if "recovered_key" in locals():
                for i in range(len(recovered_key)):
                    recovered_key[i] = 0
                del recovered_key
            if "raw_shares" in locals():
                for part in raw_shares:
                    for i in range(len(part)):
                        part[i] = 0

    def verify_shares(
        self,
        vault: EncryptedVault,
        shares: List[str],
    ) -> bool:
        """Verify whether a set of shares can successfully authenticate and unlock the vault."""
        try:
            self.recover_mapping(vault, shares)
            return True
        except VaultError:
            return False


# Singleton default vault manager
default_vault_manager = VaultManager()


def restore_text(
    redacted_text: str,
    vault: EncryptedVault,
    shares: List[str],
    vault_manager: Optional[VaultManager] = None,
) -> str:
    """Explicitly and locally restore placeholders in redacted_text using threshold shares.

    This operation is intentionally local and explicit:
    - It must never be invoked automatically on arbitrary or untrusted model outputs.
    - It strictly requires sufficient threshold shares to reconstruct the master key.
    - Master key and shares are never transmitted externally or leaked to logs.

    Args:
        redacted_text: The redacted text containing typed placeholders.
        vault: The EncryptedVault containing the authenticated ciphertext.
        shares: The threshold shares required to decrypt the vault.
        vault_manager: Optional VaultManager instance (uses default if None).

    Returns:
        The restored string with placeholders replaced by original values.
    """
    mgr = vault_manager or default_vault_manager
    decrypted_mapping = mgr.recover_mapping(vault, shares)
    restored = redacted_text
    for placeholder, original_val in decrypted_mapping.items():
        restored = restored.replace(placeholder, original_val)
    return restored

