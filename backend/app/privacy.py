"""
CipherMind Privacy Engine
=========================

Core privacy gateway module that detects supported sensitive data (emails,
phone numbers, API keys / bearer tokens, and explicitly assigned passwords/secrets)
and replaces them with typed placeholders before any external model interaction.

Supported Categories & Patterns
--------------------------------
1. EMAIL:
   - Standard email address formats matching RFC 5322 compliant regex patterns.
   - Placeholder format: [EMAIL_1], [EMAIL_2], ...

2. PHONE:
   - Conservative matching rules for international (E.164) and standard national
     telephone numbers (e.g., +1-555-123-4567, (555) 123-4567, +44 20 7946 0958).
   - Conservative boundaries prevent false positives on dates (YYYY-MM-DD),
     IP addresses, fractions, and general integer sequences.
   - Placeholder format: [PHONE_1], [PHONE_2], ...

3. API_KEY:
   - Common provider formats (OpenAI sk-*, GitHub ghp_*, Slack xox*, AWS AKIA*, Google AIza*, JWTs).
   - Bearer authentication tokens (e.g., "Bearer <token>").
   - Generic key assignments (e.g., api_key=..., access_token=..., secret_key=...).
   - Placeholder format: [API_KEY_1], [API_KEY_2], ...

4. PASSWORD:
   - Explicitly assigned passwords and credentials (e.g., password=..., passwd: ...,
     pwd="...", client_secret=..., db_pass=...).
   - Placeholder format: [PASSWORD_1], [PASSWORD_2], ...

Known Limitations & Boundaries
------------------------------
- Detection relies on regex pattern heuristics and syntax conventions.
- Unstructured passwords mentioned in natural prose without explicit assignment keywords
  (e.g., "The phrase I use to log in is sunflowermoon") cannot be reliably detected by regex.
- Novel, encrypted, or obfuscated tokens (e.g., base64-encoded or split across lines) are not matched.
- Replacement with placeholders redacts direct secret identifiers but does not sanitize
  indirect semantic context (e.g., unique organizational titles or confidential background details).
- This redaction engine does not provide homomorphic encryption or zero-knowledge guarantees.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set, Tuple


# Maximum allowed prompt size for inspection (50,000 characters / ~12,000 tokens)
MAX_TEXT_LENGTH = 50_000


@dataclass(frozen=True)
class FindingMetadata:
    """Metadata representing a detected sensitive entity category, placeholder, and occurrence count."""
    category: str
    placeholder: str
    count: int


@dataclass
class RedactionResult:
    """Result returned by the privacy engine after scanning and redacting text."""
    redacted_text: str
    findings: List[FindingMetadata]
    total_findings: int
    vault_mapping: Dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def entities(self) -> List[FindingMetadata]:
        """Alias for findings list."""
        return self.findings


@dataclass
class _CandidateMatch:
    start: int
    end: int
    category: str
    secret_value: str
    priority: int


class PrivacyEngine:
    """Engine for identifying sensitive entities in prompts and replacing them with typed placeholders."""

    # 1. Email pattern
    EMAIL_PATTERN = re.compile(
        r"(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    # 2. Conservative Phone pattern (supports US formatted, international formatted, and E.164)
    # Does not match dates (e.g. 2026-10-25) or IP addresses (e.g. 192.168.1.1)
    PHONE_PATTERN = re.compile(
        r"(?<!\w)(?:(?:\+1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]\d{4}|\+\d{1,3}[-.\s]\d{1,4}[-.\s]\d{3,4}[-.\s]\d{3,4}|\+\d{10,15})(?!\w)"
    )

    # 3. Bearer token pattern
    BEARER_PATTERN = re.compile(
        r"(?i)\bBearer\s+([A-Za-z0-9_\-\.]{16,})\b"
    )

    # 4. Known API key formats & tokens
    PROVIDER_KEY_PATTERNS = [
        re.compile(r"\b(sk-(?:proj-)?[A-Za-z0-9_\-]{20,})\b"),                         # OpenAI
        re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{36,})\b"),                               # GitHub
        re.compile(r"\b(xox[baprs]-[0-9A-Za-z]{10,}(?:-[0-9A-Za-z]{10,})*)\b"),       # Slack
        re.compile(r"\b((?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16})\b"),                     # AWS Access Key
        re.compile(r"\b(AIza[0-9A-Za-z\-_]{35})\b"),                                   # Google API Key
        re.compile(r"\b(ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"),  # JWT Token
    ]

    # 5. Generic API Key / Token assignment pattern
    GENERIC_KEY_PATTERN = re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key|token)\s*[:=]\s*(?:([\"'])(?P<qval>[^\r\n]+?)\1|(?P<uval>[A-Za-z0-9_\-]{16,}))"
    )

    # 6. Explicit password / credential assignment pattern
    PASSWORD_ASSIGN_PATTERN = re.compile(
        r"(?i)\b(?:password|passwd|pwd|passphrase|secret|client[_-]?secret|db_pass(?:word)?)\s*[:=]\s*(?:([\"'])(?P<qval>[^\r\n]+?)\1|(?P<uval>[^\s'\";,\r\n]+))"
    )

    def redact(self, text: str) -> RedactionResult:
        """Scan input text, identify sensitive patterns, and replace them with typed placeholders.

        Args:
            text: The raw input string to inspect.

        Returns:
            RedactionResult containing the redacted text, metadata of findings, and total count.

        Raises:
            ValueError: If text length exceeds MAX_TEXT_LENGTH.
        """
        if not text:
            return RedactionResult(redacted_text="", findings=[], total_findings=0)

        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Input text length ({len(text)}) exceeds maximum allowed limit of {MAX_TEXT_LENGTH} characters."
            )

        raw_candidates: List[_CandidateMatch] = []

        # 1. Collect password / credential assignments (Priority: 4)
        for m in self.PASSWORD_ASSIGN_PATTERN.finditer(text):
            val = m.group("qval") or m.group("uval")
            if val:
                span = m.span("qval") if m.group("qval") is not None else m.span("uval")
                raw_candidates.append(_CandidateMatch(span[0], span[1], "PASSWORD", val, 4))

        # 2. Collect generic API key assignments (Priority: 3)
        for m in self.GENERIC_KEY_PATTERN.finditer(text):
            val = m.group("qval") or m.group("uval")
            if val:
                span = m.span("qval") if m.group("qval") is not None else m.span("uval")
                raw_candidates.append(_CandidateMatch(span[0], span[1], "API_KEY", val, 3))

        # 3. Collect Bearer tokens (Priority: 3)
        for m in self.BEARER_PATTERN.finditer(text):
            val = m.group(1)
            raw_candidates.append(_CandidateMatch(m.start(1), m.end(1), "API_KEY", val, 3))

        # 4. Collect provider API keys (Priority: 3)
        for pat in self.PROVIDER_KEY_PATTERNS:
            for m in pat.finditer(text):
                val = m.group(1)
                raw_candidates.append(_CandidateMatch(m.start(1), m.end(1), "API_KEY", val, 3))

        # 5. Collect emails (Priority: 2)
        for m in self.EMAIL_PATTERN.finditer(text):
            raw_candidates.append(_CandidateMatch(m.start(), m.end(), "EMAIL", m.group(0), 2))

        # 6. Collect phone numbers (Priority: 1)
        for m in self.PHONE_PATTERN.finditer(text):
            raw_candidates.append(_CandidateMatch(m.start(), m.end(), "PHONE", m.group(0), 1))

        # Resolve overlapping spans: higher priority first, then longer match, then earlier start
        sorted_candidates = sorted(
            raw_candidates,
            key=lambda c: (-c.priority, -(c.end - c.start), c.start)
        )
        accepted_matches: List[_CandidateMatch] = []
        for cand in sorted_candidates:
            has_overlap = any(
                max(cand.start, acc.start) < min(cand.end, acc.end)
                for acc in accepted_matches
            )
            if not has_overlap:
                accepted_matches.append(cand)

        # Sort accepted matches in chronological text appearance
        accepted_matches.sort(key=lambda c: c.start)

        # Assign consistent typed placeholders to unique secret values per category
        category_counters: Dict[str, int] = {}
        secret_to_placeholder: Dict[Tuple[str, str], str] = {}

        for m in accepted_matches:
            key = (m.category, m.secret_value)
            if key not in secret_to_placeholder:
                current_count = category_counters.get(m.category, 0) + 1
                category_counters[m.category] = current_count
                secret_to_placeholder[key] = f"[{m.category}_{current_count}]"

        # Prepare spans for replacement: start, end, placeholder, category
        final_spans: List[Tuple[int, int, str, str]] = [
            (
                m.start,
                m.end,
                secret_to_placeholder[(m.category, m.secret_value)],
                m.category
            )
            for m in accepted_matches
        ]

        # Scan for repeated occurrences of detected secret values (>= 4 chars) across the input
        for (category, secret_val), placeholder in secret_to_placeholder.items():
            if len(secret_val) >= 4:
                # Use bounded pattern to avoid corrupting partial words
                escaped_val = re.escape(secret_val)
                rep_pattern = re.compile(rf"(?<!\w){escaped_val}(?!\w)")
                for rep_match in rep_pattern.finditer(text):
                    s, e = rep_match.start(), rep_match.end()
                    # Only add if not overlapping with an existing span
                    if not any(max(s, span[0]) < min(e, span[1]) for span in final_spans):
                        final_spans.append((s, e, placeholder, category))

        # Sort spans in reverse order of start index to safely perform replacements
        final_spans.sort(key=lambda s: s[0], reverse=True)

        # Calculate counts per placeholder and record category
        placeholder_counts: Dict[str, int] = {}
        placeholder_categories: Dict[str, str] = {}
        for _, _, placeholder, category in final_spans:
            placeholder_counts[placeholder] = placeholder_counts.get(placeholder, 0) + 1
            placeholder_categories[placeholder] = category

        # Execute text redaction
        redacted_text = text
        for s, e, placeholder, _ in final_spans:
            redacted_text = redacted_text[:s] + placeholder + redacted_text[e:]

        # Build diagnostic findings metadata (NEVER includes raw secret values)
        findings: List[FindingMetadata] = []
        sorted_placeholders = sorted(
            placeholder_counts.keys(),
            key=lambda p: (placeholder_categories[p], p)
        )
        for p in sorted_placeholders:
            findings.append(
                FindingMetadata(
                    category=placeholder_categories[p],
                    placeholder=p,
                    count=placeholder_counts[p]
                )
            )

        total_findings = sum(placeholder_counts.values())
        vault_mapping = {
            placeholder: secret_val
            for (_, secret_val), placeholder in secret_to_placeholder.items()
        }

        return RedactionResult(
            redacted_text=redacted_text,
            findings=findings,
            total_findings=total_findings,
            vault_mapping=vault_mapping,
        )


# Default singleton instance
default_privacy_engine = PrivacyEngine()


def redact_sensitive_data(text: str) -> RedactionResult:
    """Convenience function to redact sensitive entities using default PrivacyEngine."""
    return default_privacy_engine.redact(text)
