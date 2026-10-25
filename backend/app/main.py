from typing import List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.privacy import MAX_TEXT_LENGTH, default_privacy_engine
from app.vault import (
    InsufficientSharesError,
    InvalidShareError,
    VaultAuthenticationError,
    VaultNotFoundError,
    default_vault_manager,
)


class ServiceInfo(BaseModel):
    name: str
    description: str
    status: str


class HealthStatus(BaseModel):
    status: str
    service: str


class PrivacyInspectRequest(BaseModel):
    text: str = Field(
        ...,
        description="The prompt text to inspect and redact.",
    )


class Finding(BaseModel):
    category: str
    placeholder: str
    count: int


class PrivacyInspectResponse(BaseModel):
    redacted_text: str
    findings: List[Finding]
    total_findings: int
    entities: List[Finding] = Field(
        default_factory=list,
        description="Alias for findings list.",
    )


class VaultMetadataResponse(BaseModel):
    vault_id: str
    status: str
    threshold: int
    total_shares: int
    placeholders: List[str]
    created_at: float


class VaultVerifyRequest(BaseModel):
    vault_id: str = Field(..., min_length=1, max_length=64, description="ID of the vault to verify.")
    shares: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Threshold shares to verify (bounded to prevent DoS).",
    )


class VaultVerifyResponse(BaseModel):
    vault_id: str
    status: str
    threshold_met: bool
    placeholder_count: int


app = FastAPI(
    title="CipherMind API",
    description="Privacy-focused AI platform backend API.",
    version="0.1.0",
)


@app.get("/", response_model=ServiceInfo)
def read_root() -> ServiceInfo:
    """Return a short service description."""
    return ServiceInfo(
        name="CipherMind API",
        description="Privacy-focused AI platform backend foundation.",
        status="running",
    )


@app.get("/api/health", response_model=HealthStatus)
def health_check() -> HealthStatus:
    """Return health status of the service."""
    return HealthStatus(
        status="healthy",
        service="CipherMind API",
    )


@app.post(
    "/api/privacy/inspect",
    response_model=PrivacyInspectResponse,
    summary="Inspect and redact sensitive data from text",
)
def inspect_text(request: PrivacyInspectRequest) -> PrivacyInspectResponse:
    """Inspect input text, detect sensitive entities, and replace them with safe placeholders."""
    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Input text exceeds maximum allowed size of {MAX_TEXT_LENGTH} characters.",
        )

    result = default_privacy_engine.redact(request.text)
    findings_list = [
        Finding(
            category=f.category,
            placeholder=f.placeholder,
            count=f.count,
        )
        for f in result.findings
    ]

    return PrivacyInspectResponse(
        redacted_text=result.redacted_text,
        findings=findings_list,
        total_findings=result.total_findings,
        entities=findings_list,
    )


@app.get(
    "/api/vault/{vault_id}",
    response_model=VaultMetadataResponse,
    summary="Get non-sensitive metadata for an encrypted vault",
)
def get_vault_metadata(vault_id: str) -> VaultMetadataResponse:
    """Retrieve metadata for an existing encrypted vault without disclosing secrets or keys.

    Note on Access Control: Vault IDs (UUIDv4) are resource identifiers, NOT access
    control or authorization tokens. User authentication and multi-tenant access control
    are not implemented in this prototype milestone.
    """
    if len(vault_id) > 64:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vault not found.",
        )
    vault = default_vault_manager.store.get(vault_id)
    if not vault:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vault with ID '{vault_id}' not found.",
        )
    return VaultMetadataResponse(
        vault_id=vault.vault_id,
        status="encrypted",
        threshold=vault.threshold,
        total_shares=vault.total_shares,
        placeholders=vault.placeholders,
        created_at=vault.created_at,
    )


@app.post(
    "/api/vault/verify",
    response_model=VaultVerifyResponse,
    summary="Verify threshold shares against an encrypted vault",
)
def verify_vault_shares(request: VaultVerifyRequest) -> VaultVerifyResponse:
    """Verify threshold shares against an encrypted vault without returning secrets."""
    vault = default_vault_manager.store.get(request.vault_id)
    if not vault:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vault with ID '{request.vault_id}' not found.",
        )

    try:
        decrypted_mapping = default_vault_manager.recover_mapping(vault, request.shares)
        placeholder_count = len(decrypted_mapping)
        return VaultVerifyResponse(
            vault_id=vault.vault_id,
            status="authenticated",
            threshold_met=True,
            placeholder_count=placeholder_count,
        )
    except InsufficientSharesError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except InvalidShareError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except VaultAuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        )
