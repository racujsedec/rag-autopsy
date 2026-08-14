import json
from collections import Counter
from pathlib import Path

from rag_autopsy.chunking import (
    FixedSizeChunker,
    ParagraphChunker,
)
from rag_autopsy.config import ChunkingConfig
from rag_autopsy.diagnostics import (
    diagnose_chunking,
    diagnose_retrieval,
)
from rag_autopsy.evaluation import (
    recall_at_k,
    reciprocal_rank,
    resolve_ground_truth,
)
from rag_autopsy.retrieval import BM25Retriever


def load_documents():
    documents = {}

    for path in sorted(Path("data/raw").glob("*.txt")):
        documents[path.stem] = path.read_text()

    return documents


def load_questions():
    return json.loads(
        Path("data/evaluation/questions.json").read_text()
    )


def build_chunks(
    documents,
    chunker,
):
    chunks = []

    for document_id, text in documents.items():
        chunks.extend(
            chunker.chunk(
                document_id=document_id,
                text=text,
            )
        )

    return chunks


def evaluate_strategy(
    name,
    chunks,
    questions,
):
    retriever = BM25Retriever(chunks)

    recall_1_scores = []
    recall_3_scores = []
    rr_scores = []
    coverage_scores = []

    chunking_diagnoses = Counter()

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    for item in questions:
        question = item["question"]
        answerable = item.get("answerable", True)

        results = retriever.search(
            question,
            top_k=3,
        )

        top_chunk = (
            results[0].chunk.chunk_id
            if results
            else "NO RESULT"
        )

        print(f"\n{item['question_id']}")
        print(f"Question: {question}")
        print(f"Answerable: {answerable}")
        print(f"Top retrieved: {top_chunk}")

        if not answerable:
            print(
                "Evaluation: UNANSWERABLE — "
                "excluded from retrieval and "
                "chunking metrics."
            )
            continue

        evidence_text = item["evidence_text"]

        ground_truth = resolve_ground_truth(
            chunks,
            evidence_text,
        )

        relevant_chunk_ids = (
            ground_truth.relevant_chunk_ids
        )

        recall_1 = recall_at_k(
            results,
            relevant_chunk_ids,
            k=1,
        )

        recall_3 = recall_at_k(
            results,
            relevant_chunk_ids,
            k=3,
        )

        rr = reciprocal_rank(
            results,
            relevant_chunk_ids,
        )

        retrieval_diagnosis = diagnose_retrieval(
            results,
            relevant_chunk_ids,
        )

        chunking_diagnosis = diagnose_chunking(
            ground_truth,
            results,
        )

        recall_1_scores.append(recall_1)
        recall_3_scores.append(recall_3)
        rr_scores.append(rr)

        coverage_scores.append(
            ground_truth.max_coverage
        )

        chunking_diagnoses[
            chunking_diagnosis.diagnosis.value
        ] += 1

        print(
            f"Evidence coverage: "
            f"{ground_truth.max_coverage:.1%}"
        )

        print(
            "Resolved relevant chunks: "
            + ", ".join(relevant_chunk_ids)
        )

        print(
            f"Retrieval diagnosis: "
            f"{retrieval_diagnosis.diagnosis.value}"
        )

        print("\nCHUNKING AUTOPSY")
        print("-" * 40)

        print(
            f"Diagnosis: "
            f"{chunking_diagnosis.diagnosis.value}"
        )

        print(
            f"Explanation: "
            f"{chunking_diagnosis.explanation}"
        )

        print(
            f"\nRecall@1: {recall_1:.0f} | "
            f"Recall@3: {recall_3:.0f} | "
            f"RR: {rr:.3f}"
        )

    print("\nChunking diagnoses:")

    for diagnosis, count in sorted(
        chunking_diagnoses.items()
    ):
        print(f"  {diagnosis}: {count}")

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
        "avg_coverage": (
            sum(coverage_scores)
            / len(coverage_scores)
        ),
        "chunking_diagnoses": chunking_diagnoses,
    }


def main():
    documents = load_documents()
    questions = load_questions()

    fixed_chunks = build_chunks(
        documents,
        FixedSizeChunker(
            ChunkingConfig(
                chunk_size=45,
                overlap=8,
            )
        ),
    )

    paragraph_chunks = build_chunks(
        documents,
        ParagraphChunker(
            max_words=120,
        ),
    )

    fixed_metrics = evaluate_strategy(
        "FIXED-SIZE CHUNKING",
        fixed_chunks,
        questions,
    )

    paragraph_metrics = evaluate_strategy(
        "PARAGRAPH-AWARE CHUNKING",
        paragraph_chunks,
        questions,
    )

    print("\n" + "=" * 80)
    print("FINAL CHUNKING COMPARISON")
    print("=" * 80)

    print(
        f"{'Strategy':<20}"
        f"{'Recall@1':<12}"
        f"{'Recall@3':<12}"
        f"{'MRR':<10}"
        f"{'Coverage':<12}"
    )

    print("-" * 70)

    print(
        f"{'Fixed-size':<20}"
        f"{fixed_metrics['recall@1']:<12.1%}"
        f"{fixed_metrics['recall@3']:<12.1%}"
        f"{fixed_metrics['mrr']:<10.3f}"
        f"{fixed_metrics['avg_coverage']:<12.1%}"
    )

    print(
        f"{'Paragraph-aware':<20}"
        f"{paragraph_metrics['recall@1']:<12.1%}"
        f"{paragraph_metrics['recall@3']:<12.1%}"
        f"{paragraph_metrics['mrr']:<10.3f}"
        f"{paragraph_metrics['avg_coverage']:<12.1%}"
    )


if __name__ == "__main__":
    main()
