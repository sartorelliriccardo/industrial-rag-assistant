"""Small, UI-independent rules for session handling: idle expiry and the optional password."""

import hmac


def idle_expired(last_active: float, now: float, idle_minutes: float) -> bool:
    """True when the user has done nothing for longer than `idle_minutes`.

    `idle_minutes <= 0` disables the expiry. Times are seconds (e.g. time.time()).
    """
    if idle_minutes <= 0:
        return False
    return now - last_active > idle_minutes * 60


def check_interval_seconds(idle_minutes: float, default: float = 30.0) -> float:
    """How often to look for inactivity: often enough to react within a fraction of the
    timeout, but never more than every second and never more than `default`."""
    return max(1.0, min(default, idle_minutes * 60 / 2))


def passwords_match(expected: str, given: str) -> bool:
    """Constant-time comparison, so response time does not leak how much matched."""
    return hmac.compare_digest(expected.encode("utf-8"), given.encode("utf-8"))
