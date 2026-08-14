import json
from collections import Counter
from pathlib import Path

from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig
from rag_autopsy.diagnostics import diagnose_retrieval
from rag_autopsy.evaluation import recall_at_k, reciprocal_rank
from rag_autopsy.evaluation.ground_truth import resolve_ground_truth
from rag_autopsy.retrieval import BM25Retriever


def load_chunks():
    raw_dir = Path("data/raw")

    chunker = FixedSizeChunker(
        ChunkingConfig(
            chunk_size=45,
            overlap=8,
        )
    )

    all_chunks = []

    for path in sorted(raw_dir.glob("*.txt")):
        text = path.read_text()

        chunks = chunker.chunk(
            document_id=path.stem,
            text=text,
        )

        all_chunks.extend(chunks)

    return all_chunks


def load_questions():
    path = Path("data/evaluation/questions.json")
    return json.loads(path.read_text())


def main():
    chunks = load_chunks()
    questions = load_questions()

    retriever = BM25Retriever(chunks)

    recall_1_scores = []
    recall_3_scores = []
    reciprocal_rank_scores = []
    diagnoses = []

    print("\nRAG AUTOPSY — BM25 RETRIEVAL EVALUATION")
    print("=" * 80)

    for item in questions:
        question = item["question"]
        evidence_text = item["evidence_text"]

        ground_truth = resolve_ground_truth(
            chunks=chunks,
            evidence_text=evidence_text,
        )

        relevant_chunk_ids = ground_truth.relevant_chunk_ids

        results = retriever.search(
            query=question,
            top_k=3,
        )

        recall_1 = recall_at_k(
            results,
            relevant_chunk_ids=relevant_chunk_ids,
            k=1,
        )

        recall_3 = recall_at_k(
            results,
            relevant_chunk_ids=relevant_chunk_ids,
            k=3,
        )

        rr = reciprocal_rank(
            results,
            relevant_chunk_ids=relevant_chunk_ids,
        )

        autopsy = diagnose_retrieval(
            results,
            relevant_chunk_ids,
        )

        recall_1_scores.append(recall_1)
        recall_3_scores.append(recall_3)
        reciprocal_rank_scores.append(rr)
        diagnoses.append(autopsy.diagnosis.value)

        print(f"\n{item['question_id']}")
        print(f"Question: {question}")
        print(
            f"Relevant chunks: "
            f"{', '.join(relevant_chunk_ids)}"
        )

        print(
            f"Evidence coverage: "
            f"{ground_truth.max_coverage:.1%}"
        )

        print(
            f"Complete evidence preserved: "
            f"{ground_truth.complete_evidence_preserved}"
        )

        print("\nTop results:")

        if not results:
            print("  No matching chunks retrieved.")

        for rank, result in enumerate(results, start=1):
            marker = (
                "✅"
                if result.chunk.chunk_id in relevant_chunk_ids
                else "❌"
            )

            print(
                f"  #{rank} {marker} "
                f"{result.chunk.chunk_id} "
                f"(score={result.score:.4f})"
            )

        print("\nAUTOPSY")
        print("-" * 40)
        print(f"Diagnosis: {autopsy.diagnosis.value}")
        print(f"Relevant rank: {autopsy.relevant_rank}")
        print(f"Explanation: {autopsy.explanation}")

        print("\nMetrics")
        print(f"Recall@1: {recall_1:.0f}")
        print(f"Recall@3: {recall_3:.0f}")
        print(f"Reciprocal Rank: {rr:.3f}")

    average_recall_1 = (
        sum(recall_1_scores) / len(recall_1_scores)
    )

    average_recall_3 = (
        sum(recall_3_scores) / len(recall_3_scores)
    )

    mean_reciprocal_rank = (
        sum(reciprocal_rank_scores)
        / len(reciprocal_rank_scores)
    )

    diagnosis_counts = Counter(diagnoses)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Questions evaluated: {len(questions)}")
    print(f"Recall@1: {average_recall_1:.1%}")
    print(f"Recall@3: {average_recall_3:.1%}")
    print(f"MRR: {mean_reciprocal_rank:.3f}")

    print("\nAutopsy Results:")

    for diagnosis, count in sorted(diagnosis_counts.items()):
        print(f"  {diagnosis}: {count}")


if __name__ == "__main__":
    main()
