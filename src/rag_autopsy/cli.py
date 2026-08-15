import argparse
import json
import os
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-autopsy",
        description=(
            "RAG Autopsy diagnostic and evaluation CLI."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    autopsy_parser = subparsers.add_parser(
        "autopsy",
        help="Run a RAG autopsy for a question.",
    )

    question_group = (
        autopsy_parser.add_mutually_exclusive_group(
            required=True
        )
    )

    question_group.add_argument(
        "--question",
        help="Question to analyze.",
    )

    question_group.add_argument(
        "--question-id",
        help="Benchmark question ID to analyze.",
    )

    autopsy_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of retrieved chunks to inspect.",
    )

    return parser


def load_benchmark_chunks():
    from rag_autopsy.chunking import ParagraphChunker

    chunker = ParagraphChunker(
        max_words=120
    )

    chunks = []

    for path in sorted(
        Path("data/raw").glob("*.txt")
    ):
        chunks.extend(
            chunker.chunk(
                document_id=path.stem,
                text=path.read_text(),
            )
        )

    return chunks


def load_benchmark_questions():
    return json.loads(
        Path(
            "data/evaluation/questions.json"
        ).read_text()
    )


def run_benchmark_retrieval(
    question_id: str,
    top_k: int,
) -> str:
    import psycopg

    from rag_autopsy.evaluation import (
        resolve_ground_truth,
    )
    from rag_autopsy.retrieval import (
        PgVectorRetriever,
    )

    questions = load_benchmark_questions()

    item = next(
        (
            question
            for question in questions
            if question["question_id"]
            == question_id
        ),
        None,
    )

    if item is None:
        raise ValueError(
            f"Unknown benchmark question ID: {question_id}"
        )

    chunks = load_benchmark_chunks()

    if item.get(
        "answerable",
        True,
    ):
        ground_truth = resolve_ground_truth(
            chunks=chunks,
            evidence_text=item[
                "evidence_text"
            ],
        )

        relevant_chunk_ids = (
            ground_truth.relevant_chunk_ids
        )
    else:
        relevant_chunk_ids = ()

    database_url = os.getenv(
        "RAG_AUTOPSY_DATABASE_URL",
        "dbname=rag_autopsy",
    )

    with psycopg.connect(
        database_url
    ) as connection:
        retriever = PgVectorRetriever(
            connection=connection
        )

        results = retriever.search(
            item["question"],
            top_k=top_k,
        )

    if relevant_chunk_ids:
        from rag_autopsy.diagnostics.autopsy import (
            diagnose_retrieval,
        )

        retrieval_diagnosis = diagnose_retrieval(
            results=results,
            relevant_chunk_ids=list(
                relevant_chunk_ids
            ),
        )

        diagnosis_lines = [
            (
                "PRIMARY RETRIEVAL DIAGNOSIS: "
                f"{retrieval_diagnosis.diagnosis.value}"
            ),
            retrieval_diagnosis.explanation,
        ]
    else:
        diagnosis_lines = [
            "PRIMARY RETRIEVAL DIAGNOSIS: NOT_APPLICABLE",
            (
                "This benchmark question is marked "
                "unanswerable, so no relevant chunk "
                "is expected."
            ),
        ]

    if relevant_chunk_ids:
        from rag_autopsy.diagnostics.autopsy import (
            diagnose_context,
        )

        context_diagnosis = diagnose_context(
            question=item["question"],
            chunks=chunks,
            ground_truth=ground_truth,
            retrieval_results=results,
        )

        context_lines = [
            (
                "CONTEXT DIAGNOSIS: "
                f"{context_diagnosis.diagnosis.value}"
            ),
            context_diagnosis.explanation,
        ]
    else:
        context_lines = [
            "CONTEXT DIAGNOSIS: NOT_APPLICABLE",
            (
                "This benchmark question is marked "
                "unanswerable, so context-loss analysis "
                "is not applicable."
            ),
        ]

    lines = [
        f"Question ID: {question_id}",
        f"Question: {item['question']}",
        "",
        *diagnosis_lines,
        "",
        *context_lines,
        "",
        "Relevant chunks: "
        + (
            ", ".join(
                relevant_chunk_ids
            )
            if relevant_chunk_ids
            else "None"
        ),
        "",
        "Retrieved chunks:",
    ]

    if not results:
        lines.append("None")
    else:
        for rank, result in enumerate(
            results,
            start=1,
        ):
            lines.append(
                f"{rank}. "
                f"{result.chunk.chunk_id} "
                f"| score={result.score:.4f}"
            )

    return "\n".join(lines)


def main(
    argv: list[str] | None = None,
) -> int:
    parser = build_parser()

    if argv == []:
        parser.print_help()
        return 0

    args = parser.parse_args(argv)

    if args.command == "autopsy":
        if args.question_id:
            print(
                run_benchmark_retrieval(
                    question_id=args.question_id,
                    top_k=args.top_k,
                )
            )
        else:
            print(
                f"Question: {args.question}"
            )
            print(
                f"Top-k: {args.top_k}"
            )

        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
