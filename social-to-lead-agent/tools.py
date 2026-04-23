"""Tooling helpers for AutoStream lead capture."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
KNOWN_PLATFORMS = {
    "youtube",
    "instagram",
    "tiktok",
    "twitch",
    "facebook",
    "x",
    "twitter",
    "linkedin",
    "snapchat",
}


def _clean_text(value: Optional[str]) -> str:
    """Return a normalized string, or an empty string if value is falsy."""
    if not value:
        return ""
    return value.strip()


def _is_valid_email(email: str) -> bool:
    """Validate basic email format with a strict regex."""
    return bool(EMAIL_PATTERN.match(email.strip()))


def validate_lead_data(lead_data: Dict[str, Optional[str]]) -> bool:
    """Validate lead data before tool execution.

    The lead is considered valid only when all required fields exist and pass
    basic validation:
    - name is non-empty and at least 2 characters
    - email has a valid format
    - platform is non-empty and recognized or at least 2 characters

    Args:
        lead_data: A dictionary with potential keys name, email, and platform.

    Returns:
        True when all required values are valid, otherwise False.
    """
    name = _clean_text(lead_data.get("name"))
    email = _clean_text(lead_data.get("email"))
    platform = _clean_text(lead_data.get("platform"))

    if len(name) < 2:
        logger.debug("Lead validation failed: name too short or missing")
        return False

    if not _is_valid_email(email):
        logger.debug("Lead validation failed: invalid email format")
        return False

    if len(platform) < 2:
        logger.debug("Lead validation failed: platform too short or missing")
        return False

    if platform.lower() not in KNOWN_PLATFORMS and len(platform) < 3:
        logger.debug("Lead validation failed: unknown platform value")
        return False

    return True


def mock_lead_capture(name: str, email: str, platform: str) -> Dict[str, str]:
    """Simulate a lead capture API call.

    In production, this function would send validated lead data to a CRM or
    backend endpoint. Here, it logs and prints the values for demonstration.

    Args:
        name: Full name of the lead.
        email: Contact email address.
        platform: Primary creator platform (e.g., YouTube, Instagram).

    Returns:
        A dictionary representing a successful mock capture response.

    Raises:
        ValueError: If any input fails validation.
    """
    candidate = {
        "name": _clean_text(name),
        "email": _clean_text(email),
        "platform": _clean_text(platform),
    }

    if not validate_lead_data(candidate):
        raise ValueError("Lead data is incomplete or invalid. Capture aborted.")

    logger.info("Lead captured successfully: email=%s, platform=%s", email, platform)

    print("Lead captured successfully!")
    print(f"Name: {candidate['name']}")
    print(f"Email: {candidate['email']}")
    print(f"Platform: {candidate['platform']}")

    return {
        "status": "success",
        "lead_id": "mock_id_12345",
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }
