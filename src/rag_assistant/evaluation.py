"""Repeatable evaluation: retrieval, answer correctness, citation accuracy, abstention.

    python -m rag_assistant.evaluation --docs data/sample_pdfs --questions eval/questions.jsonl

Runs offline by default (local embedding model + extractive provider, no API key).
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from rag_assistant.config import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    MIN_SCORE_BY_SEARCH,
    get_chunk_settings,
)
from rag_assistant.embeddings import (
    Embedder,
    FastEmbedEmbedder,
    HashingEmbedder,
    LexicalEmbedder,
)
from rag_assistant.llm import get_provider
from rag_assistant.pipeline import RagPipeline


@dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    answerable: bool
    status: str
    answer: str
    cited: list[str]
    retrieved: list[str]
    hit_rank: int | None = None  # rank of first gold page in retrieved, None = miss
    gold_recall: float | None = None  # share of the gold pages found among the retrieved ones
    correct: bool | None = None
    citation_hit: bool | None = None
    citation_precision: float | None = None
    abstained: bool = False
    problems: list[str] = field(default_factory=list)


def load_questions(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _page_key(doc: str, page: int) -> str:
    return f"{doc}#p{page}"


def evaluate_question(pipeline: RagPipeline, q: dict) -> QuestionResult:
    retrieved = pipeline.search(q["question"])
    answer = pipeline.ask(q["question"])
    retrieved_keys = [_page_key(r.chunk.doc_name, r.chunk.page_number) for r in retrieved]
    cited_keys = [_page_key(c.doc_name, c.page_number) for c in answer.citations]
    res = QuestionResult(
        id=q["id"],
        category=q["category"],
        question=q["question"],
        answerable=q["answerable"],
        status=answer.status,
        answer=answer.text,
        cited=cited_keys,
        retrieved=retrieved_keys,
        abstained=answer.status == "insufficient",
    )

    if not q["answerable"]:
        if not res.abstained:
            res.problems.append("answered a question that has no answer in the documents")
        return res

    gold = {_page_key(g["document"], g["page"]) for g in q["gold"]}
    res.hit_rank = next((i for i, k in enumerate(retrieved_keys, 1) if k in gold), None)
    res.gold_recall = len(gold & set(retrieved_keys)) / len(gold) if gold else None
    if res.hit_rank is None:
        res.problems.append("retrieval miss: no gold page in top-k")

    if res.abstained:
        res.correct = False
        res.problems.append("false abstention")
        return res

    text = answer.text.lower()
    # every group of alternatives must be matched by at least one of its members
    groups = q.get("expected_any_of") or [[k] for k in q.get("expected_keywords", [])]
    has_all = all(any(alt.lower() in text for alt in group) for group in groups)
    has_forbidden = any(k.lower() in text for k in q.get("forbidden_keywords", []))
    res.correct = has_all and not has_forbidden and answer.status == "answered"
    if not has_all:
        res.problems.append("missing expected keyword(s)")
    if has_forbidden:
        res.problems.append("contains forbidden content (injection followed?)")

    unique_cited = set(cited_keys)
    res.citation_hit = bool(unique_cited & gold)
    res.citation_precision = len(unique_cited & gold) / len(unique_cited) if unique_cited else 0.0
    if not res.citation_hit:
        res.problems.append("no gold page among citations")
    return res


def summarize(results: list[QuestionResult]) -> dict:
    ans = [r for r in results if r.answerable]
    unans = [r for r in results if not r.answerable]
    answered = [r for r in ans if not r.abstained]

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 3) if den else None

    return {
        "n_questions": len(results),
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        "retrieval_hit_at_1": rate(sum(r.hit_rank == 1 for r in ans), len(ans)),
        "retrieval_hit_at_3": rate(
            sum(r.hit_rank is not None and r.hit_rank <= 3 for r in ans), len(ans)
        ),
        "retrieval_hit_at_k": rate(sum(r.hit_rank is not None for r in ans), len(ans)),
        # questions whose answer needs several pages: were ALL of them retrieved?
        "retrieval_all_gold_pages": rate(sum(r.gold_recall == 1.0 for r in ans), len(ans)),
        "retrieval_mrr": round(sum(1 / r.hit_rank for r in ans if r.hit_rank) / len(ans), 3)
        if ans
        else None,
        "answer_correct": rate(sum(bool(r.correct) for r in ans), len(ans)),
        "citation_hit": rate(sum(bool(r.citation_hit) for r in answered), len(answered)),
        "citation_precision": round(
            sum(r.citation_precision or 0 for r in answered) / len(answered), 3
        )
        if answered
        else None,
        "abstention_correct": rate(sum(r.abstained for r in unans), len(unans)),
        "false_abstention": rate(sum(r.abstained for r in ans), len(ans)),
    }


def run_evaluation(
    docs_dir: str | Path,
    questions_path: str | Path,
    embedder: Embedder,
    provider_name: str = "extractive",
    chunk_size: int | None = None,
    overlap: int | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
) -> tuple[dict, list[QuestionResult]]:
    pipeline = RagPipeline(
        embedder,
        get_provider(provider_name),
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        top_k=top_k,
        min_score=min_score,
    )
    for pdf in sorted(Path(docs_dir).glob("*.pdf")):
        pipeline.add_pdf_path(pdf)
    results = [evaluate_question(pipeline, q) for q in load_questions(questions_path)]
    summary = summarize(results)
    summary["config"] = {
        "search": type(embedder).__name__,
        "provider": provider_name,
        "chunk_size": pipeline.chunk_size,
        "chunk_overlap": pipeline.chunk_overlap,
        "top_k": top_k,
        "min_score": min_score,
        "n_chunks": pipeline.chunk_count,
    }
    return summary, results


def _print_summary(summary: dict) -> None:
    print(f"\nconfig: {summary['config']}")
    for key, value in summary.items():
        if key != "config":
            print(f"  {key:<22} {value}")


def main() -> None:
    default_size, default_overlap = get_chunk_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--docs", default="data/sample_pdfs")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument(
        "--search",
        choices=["keyword", "semantic", "hash"],
        default="keyword",
        help="keyword = stemmed words, no model; semantic = neural embeddings; hash = test only",
    )
    p.add_argument("--provider", choices=["extractive", "anthropic"], default="extractive")
    p.add_argument("--chunk-size", type=int, default=default_size)
    p.add_argument("--overlap", type=int, default=default_overlap)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="minimum similarity (default depends on --search)",
    )
    p.add_argument("--sweep-min-score", type=float, nargs="*", help="try several thresholds")
    p.add_argument(
        "--baseline", action="store_true", help="also report what plain Ctrl+F would find"
    )
    p.add_argument("--out", help="write full JSON results here")
    args = p.parse_args()

    embedder = {
        "keyword": LexicalEmbedder,
        "semantic": FastEmbedEmbedder,
        "hash": HashingEmbedder,
    }[args.search]()
    if args.min_score is None:
        args.min_score = MIN_SCORE_BY_SEARCH.get(args.search, DEFAULT_MIN_SCORE)
    thresholds = args.sweep_min_score or [args.min_score]
    last_results: list[QuestionResult] = []
    all_summaries = []
    for threshold in thresholds:
        summary, last_results = run_evaluation(
            args.docs,
            args.questions,
            embedder,
            args.provider,
            args.chunk_size,
            args.overlap,
            args.top_k,
            threshold,
        )
        _print_summary(summary)
        all_summaries.append(summary)

    if args.baseline:
        from rag_assistant.baselines import evaluate_ctrl_f

        print("\nCtrl+F baseline (exact words, retrieval only):")
        for mode in ("all", "rarest"):
            result = evaluate_ctrl_f(args.docs, load_questions(args.questions), mode)
            print(f"  {result}")
            all_summaries.append({"baseline": result})

    print("\nProblems (last configuration):")
    for r in last_results:
        for problem in r.problems:
            print(f"  {r.id} [{r.category}] {r.question}\n      -> {problem}")

    if args.out:
        payload = {
            "summaries": all_summaries,
            "results": [r.__dict__ for r in last_results],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
