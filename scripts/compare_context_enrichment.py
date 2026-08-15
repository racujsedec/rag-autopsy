import json
from pathlib import Path

from rag_autopsy.chunking import (
    ParagraphChunker,
    PreviousChunkContextEnricher,
)
from rag_autopsy.evaluation import (
    recall_at_k,
    reciprocal_rank,
    resolve_ground_truth,
)
from rag_autopsy.retrieval import SemanticRetriever


def load_chunks():
    raw_dir = Path("data/raw")

    chunker = ParagraphChunker(
        max_words=120
    )

    chunks = []

    for path in sorted(raw_dir.glob("*.txt")):
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


def find_rank(
    results,
    relevant_chunk_ids,
):
    relevant = set(
        relevant_chunk_ids
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result.chunk.chunk_id in relevant:
            return rank

    return None


def evaluate(
    retriever,
    questions,
    ground_truth_chunks,
):
    recall_1_scores = []
    recall_3_scores = []
    rr_scores = []
    ranks = {}

    for item in questions:
        if not item.get(
            "answerable",
            True,
        ):
            continue

        ground_truth = resolve_ground_truth(
            chunks=ground_truth_chunks,
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

        ranks[item["question_id"]] = (
            find_rank(
                results,
                relevant,
            )
        )

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
        "ranks": ranks,
    }


def rank_label(rank):
    if rank is None:
        return "MISS"

    return str(rank)


def main():
    original_chunks = load_chunks()
    questions = load_questions()

    enricher = (
        PreviousChunkContextEnricher()
    )

    contextual_chunks = enricher.enrich(
        original_chunks
    )

    print(
        "\nLoading original semantic retriever..."
    )
    original_retriever = SemanticRetriever(
        original_chunks
    )

    print(
        "Loading context-enriched "
        "semantic retriever..."
    )
    contextual_retriever = (
        SemanticRetriever(
            contextual_chunks
        )
    )

    original_metrics = evaluate(
        original_retriever,
        questions,
        original_chunks,
    )

    contextual_metrics = evaluate(
        contextual_retriever,
        questions,
        original_chunks,
    )

    print("\n" + "=" * 80)
    print(
        "SEMANTIC CONTEXT-ENRICHMENT "
        "RANK CHANGES"
    )
    print("=" * 80)

    change_count = 0

    for item in questions:
        if not item.get(
            "answerable",
            True,
        ):
            continue

        question_id = item["question_id"]

        before = original_metrics[
            "ranks"
        ][question_id]

        after = contextual_metrics[
            "ranks"
        ][question_id]

        if before == after:
            continue

        change_count += 1

        print(
            f"{question_id} | "
            f"{item['question_type']:<14} | "
            f"{rank_label(before):>4} "
            f"-> "
            f"{rank_label(after):<4}"
        )

    if change_count == 0:
        print("No rank changes.")

    print("\n" + "=" * 80)
    print(
        "FINAL CONTEXT-ENRICHMENT "
        "COMPARISON"
    )
    print("=" * 80)

    print(
        f"{'Configuration':<24}"
        f"{'Recall@1':<15}"
        f"{'Recall@3':<15}"
        f"{'MRR':<15}"
    )

    print("-" * 69)

    print(
        f"{'Semantic original':<24}"
        f"{original_metrics['recall@1']:<15.1%}"
        f"{original_metrics['recall@3']:<15.1%}"
        f"{original_metrics['mrr']:<15.3f}"
    )

    print(
        f"{'Semantic contextual':<24}"
        f"{contextual_metrics['recall@1']:<15.1%}"
        f"{contextual_metrics['recall@3']:<15.1%}"
        f"{contextual_metrics['mrr']:<15.3f}"
    )


if __name__ == "__main__":
    main()
