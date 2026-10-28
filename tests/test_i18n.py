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


def test_problem_texts_and_search_modes_have_translations():
    from rag_assistant.i18n import PROBLEM_KEYS

    for lang in LANGUAGES:
        for key in [*PROBLEM_KEYS.values(), "search_keyword", "search_semantic"]:
            assert key in STRINGS[lang], f"{lang}: missing {key}"


def test_every_problem_the_evaluation_can_report_is_mapped():
    from pathlib import Path

    from rag_assistant.i18n import PROBLEM_KEYS

    source = (Path(__file__).parent.parent / "src/rag_assistant/evaluation.py").read_text("utf-8")
    for problem in PROBLEM_KEYS:
        assert problem in source, f"{problem!r} is not an evaluation message any more"


def test_default_language_is_supported():
    assert DEFAULT_LANGUAGE in LANGUAGES
