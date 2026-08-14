import json
from collections import Counter
from pathlib import Path

from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig
from rag_autopsy.diagnostics import (
    compare_reranking_stages,
)
from rag_autopsy.evaluation.ground_truth import resolve_ground_truth
from rag_autopsy.retrieval import (
    HybridRetriever,
    RerankingRetriever,
)


def load_chunks():
    chunker = FixedSizeChunker(
        ChunkingConfig(
            chunk_size=45,
            overlap=8,
        )
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


def main():
    chunks = load_chunks()
    questions = load_questions()

    print("Loading hybrid retrieval...")
    hybrid = HybridRetriever(chunks)

    print("Loading cross-encoder reranker...")
    reranker = RerankingRetriever(
        base_retriever=hybrid,
        candidate_k=6,
    )

    diagnosis_counts = Counter()

    print(
        "\nRAG AUTOPSY — RERANKER STAGE ANALYSIS"
    )

    print("=" * 80)

    for item in questions:
        question = item["question"]
        answerable = item.get("answerable", True)

        before_results = hybrid.search(
            question,
            top_k=6,
        )

        after_results = reranker.search(
            question,
            top_k=6,
        )

        print(f"\n{item['question_id']}")
        print(f"Question: {question}")
        print(f"Answerable: {answerable}")

        if not answerable:
            before_top = (
                before_results[0].chunk.chunk_id
                if before_results
                else "NO RESULT"
            )

            after_top = (
                after_results[0].chunk.chunk_id
                if after_results
                else "NO RESULT"
            )

            print(f"Before reranking top: {before_top}")
            print(f"After reranking top: {after_top}")

            print(
                "Evaluation: UNANSWERABLE — "
                "excluded from reranker stage analysis."
            )
            continue

        evidence_text = item["evidence_text"]

        ground_truth = resolve_ground_truth(
            chunks=chunks,
            evidence_text=evidence_text,
        )

        relevant = ground_truth.relevant_chunk_ids

        comparison = compare_reranking_stages(
            before_results,
            after_results,
            relevant,
        )

        diagnosis_counts[
            comparison.diagnosis.value
        ] += 1

        print(
            "Relevant chunks: "
            + ", ".join(relevant)
        )

        print(
            f"Evidence coverage: "
            f"{ground_truth.max_coverage:.1%}"
        )

        print(
            f"Complete evidence preserved: "
            f"{ground_truth.complete_evidence_preserved}"
        )

        print(
            f"Before reranking rank: "
            f"{comparison.before_rank}"
        )

        print(
            f"After reranking rank: "
            f"{comparison.after_rank}"
        )

        print(
            f"Pipeline diagnosis: "
            f"{comparison.diagnosis.value}"
        )

        print(
            f"Explanation: "
            f"{comparison.explanation}"
        )

    print("\n" + "=" * 80)
    print("STAGE ANALYSIS SUMMARY")
    print("=" * 80)

    for diagnosis, count in sorted(
        diagnosis_counts.items()
    ):
        print(
            f"{diagnosis}: {count}"
        )


if __name__ == "__main__":
    main()
