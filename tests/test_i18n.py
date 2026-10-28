import string

import pytest

from rag_assistant.i18n import DEFAULT_LANGUAGE, LANGUAGES, STRINGS, t


def _fields(text: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_every_language_has_the_same_keys():
    reference = set(STRINGS["en"])
    for lang in LANGUAGES:
        assert set(STRINGS[lang]) == reference, f"{lang} keys differ from English"


@pytest.mark.parametrize("key", sorted(STRINGS["en"]))
def test_placeholders_match_across_languages(key):
    assert _fields(STRINGS["it"][key]) == _fields(STRINGS["en"][key])


def test_translation_fills_placeholders():
    assert "a.pdf" in t("it", "indexed", name="a.pdf", n=3)
    assert t("en", "remove") != t("it", "remove")


def test_unknown_language_falls_back_to_english():
    assert t("xx", "remove") == t("en", "remove")


def test_default_language_is_supported():
    assert DEFAULT_LANGUAGE in LANGUAGES
