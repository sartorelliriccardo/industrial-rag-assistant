import pytest

from rag_assistant.session import check_interval_seconds, idle_expired, passwords_match


def test_not_expired_before_the_limit():
    assert not idle_expired(last_active=1000, now=1000 + 14 * 60, idle_minutes=15)


def test_expired_after_the_limit():
    assert idle_expired(last_active=1000, now=1000 + 15 * 60 + 1, idle_minutes=15)


@pytest.mark.parametrize("minutes", [0, -5])
def test_non_positive_limit_disables_expiry(minutes):
    assert not idle_expired(last_active=0, now=10**9, idle_minutes=minutes)


@pytest.mark.parametrize(
    ("idle_minutes", "expected"),
    [(15, 30.0), (1, 30.0), (0.5, 15.0), (0.1, 3.0), (0.001, 1.0)],
)
def test_check_interval_is_bounded(idle_minutes, expected):
    assert check_interval_seconds(idle_minutes) == pytest.approx(expected)


def test_password_comparison():
    assert passwords_match("s3cret", "s3cret")
    assert not passwords_match("s3cret", "s3creT")
    assert not passwords_match("s3cret", "")
    assert passwords_match("pässwörd", "pässwörd")
