"""The Evaluation tab: results of the repeatable evaluation, read from eval/results_*.json."""

import json
from pathlib import Path

import streamlit as st

from rag_assistant.i18n import PROBLEM_KEYS, t

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def render_evaluation(lang: str, search: str) -> None:
    st.subheader(t(lang, "eval_title"))
    results_path = EVAL_DIR / f"results_{search}.json"  # the file matching the chosen search
    if not results_path.exists():
        st.info(t(lang, "eval_missing"))
        return
    data = json.loads(results_path.read_text(encoding="utf-8"))
    summary, results = data["summaries"][-1], data["results"]
    cfg = summary["config"]
    st.caption(
        t(
            lang,
            "eval_intro",
            mode=t(lang, f"search_{search}"),
            n=summary["n_questions"],
            n_ans=summary["n_answerable"],
            n_unans=summary["n_unanswerable"],
            size=cfg["chunk_size"],
            overlap=cfg["chunk_overlap"],
            k=cfg["top_k"],
            min_score=cfg["min_score"],
        )
    )
    metrics = [
        ("m_hit1", "retrieval_hit_at_1"),
        ("m_hit3", "retrieval_hit_at_3"),
        ("m_mrr", "retrieval_mrr"),
        ("m_correct", "answer_correct"),
        ("m_cit_hit", "citation_hit"),
        ("m_cit_prec", "citation_precision"),
        ("m_abst", "abstention_correct"),
        ("m_false_abst", "false_abstention"),
    ]
    columns = st.columns(4)
    for i, (label_key, field) in enumerate(metrics):
        value = summary.get(field)
        columns[i % 4].metric(t(lang, label_key), "-" if value is None else f"{value:.0%}")
    st.warning(t(lang, "eval_limits_md"))

    st.subheader(t(lang, "eval_problems"))
    rows = [
        {
            t(lang, "col_id"): r["id"],
            t(lang, "col_category"): r["category"],
            t(lang, "col_question"): r["question"],
            t(lang, "col_problem"): (t(lang, PROBLEM_KEYS[p]) if p in PROBLEM_KEYS else p),
        }
        for r in results
        for p in r["problems"]
    ]
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.success(t(lang, "eval_no_problems"))
