"""Pure deterministic utilities for date/time formatting and cryptographic ID hashing."""

from datetime import datetime, timezone
import uuid


def get_utc_iso_timestamp(dt: datetime | None = None) -> str:
    target = dt if dt is not None else datetime.now(timezone.utc)
    return target.isoformat()


def generate_uuidv7_str() -> str:
    return str(uuid.uuid4())
