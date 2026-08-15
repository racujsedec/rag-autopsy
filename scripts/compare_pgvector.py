import json
import os
from pathlib import Path

import psycopg

from rag_autopsy.chunking import ParagraphChunker, PreviousChunkContextEnricher
from rag_autopsy.evaluation import (
    recall_at_k,
    reciprocal_rank,
    resolve_ground_truth,
)
from rag_autopsy.retrieval import (
    PgVectorRetriever,
    SemanticRetriever,
)


def load_chunks():
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


def load_questions():
    return json.loads(
        Path(
            "data/evaluation/questions.json"
        ).read_text()
    )


def evaluate(
    retriever,
    questions,
    chunks,
):
    recall_1_scores = []
    recall_3_scores = []
    rr_scores = []
    rankings = {}

    for item in questions:
        if not item.get(
            "answerable",
            True,
        ):
            continue

        ground_truth = resolve_ground_truth(
            chunks=chunks,
            evidence_text=item["evidence_text"],
        )

        results = retriever.search(
            item["question"],
            top_k=3,
        )

        relevant = (
            ground_truth.relevant_chunk_ids
        )

        recall_1_scores.append(
            recall_at_k(
                results,
                relevant,
                k=1,
            )
        )

        recall_3_scores.append(
            recall_at_k(
                results,
                relevant,
                k=3,
            )
        )

        rr_scores.append(
            reciprocal_rank(
                results,
                relevant,
            )
        )

        rankings[item["question_id"]] = [
            result.chunk.chunk_id
            for result in results
        ]

    return {
        "recall@1": (
            sum(recall_1_scores)
            / len(recall_1_scores)
        ),
        "recall@3": (
            sum(recall_3_scores)
            / len(recall_3_scores)
        ),
        "mrr": (
            sum(rr_scores)
            / len(rr_scores)
        ),
        "rankings": rankings,
    }


def main():
    chunks = load_chunks()
    questions = load_questions()

    contextual_chunks = (
        PreviousChunkContextEnricher().enrich(
            chunks
        )
    )

    database_url = os.getenv(
        "RAG_AUTOPSY_DATABASE_URL",
        "dbname=rag_autopsy",
    )

    print(
        "\nLoading in-memory semantic retriever..."
    )

    semantic = SemanticRetriever(
        contextual_chunks
    )

    with psycopg.connect(
        database_url
    ) as connection:
        print(
            "Loading PostgreSQL pgvector retriever..."
        )

        pgvector = PgVectorRetriever(
            connection=connection
        )

        semantic_metrics = evaluate(
            semantic,
            questions,
            chunks,
        )

        pgvector_metrics = evaluate(
            pgvector,
            questions,
            chunks,
        )

    mismatches = [
        question_id
        for question_id in semantic_metrics[
            "rankings"
        ]
        if semantic_metrics[
            "rankings"
        ][question_id]
        != pgvector_metrics[
            "rankings"
        ][question_id]
    ]

    print("\n" + "=" * 80)
    print(
        "SEMANTIC VS PGVECTOR BENCHMARK"
    )
    print("=" * 80)

    print(
        f"{'Retriever':<24}"
        f"{'Recall@1':<15}"
        f"{'Recall@3':<15}"
        f"{'MRR':<15}"
    )

    print("-" * 69)

    print(
        f"{'In-memory contextual':<24}"
        f"{semantic_metrics['recall@1']:<15.1%}"
        f"{semantic_metrics['recall@3']:<15.1%}"
        f"{semantic_metrics['mrr']:<15.3f}"
    )

    print(
        f"{'PostgreSQL pgvector':<24}"
        f"{pgvector_metrics['recall@1']:<15.1%}"
        f"{pgvector_metrics['recall@3']:<15.1%}"
        f"{pgvector_metrics['mrr']:<15.3f}"
    )

    print()
    print(
        "Top-3 ranking mismatches:",
        len(mismatches),
    )

    print(
        "Questions:",
        ", ".join(mismatches)
        if mismatches
        else "None",
    )


if __name__ == "__main__":
    main()
