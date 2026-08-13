import json
from pathlib import Path

from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig
from rag_autopsy.evaluation import recall_at_k, reciprocal_rank
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

    print("\nBM25 CHUNK-LEVEL RETRIEVAL EVALUATION")
    print("=" * 80)

    for item in questions:
        question = item["question"]
        relevant_chunk_ids = item["relevant_chunk_ids"]

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

        recall_1_scores.append(recall_1)
        recall_3_scores.append(recall_3)
        reciprocal_rank_scores.append(rr)

        top_chunk = (
            results[0].chunk.chunk_id
            if results
            else "NO RESULT"
        )

        print(f"\n{item['question_id']}")
        print(f"Question: {question}")
        print(f"Relevant chunks: {', '.join(relevant_chunk_ids)}")
        print(f"Top retrieved: {top_chunk}")
        print(f"Recall@1: {recall_1:.0f}")
        print(f"Recall@3: {recall_3:.0f}")
        print(f"Reciprocal Rank: {rr:.3f}")

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

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Questions evaluated: {len(questions)}")
    print(f"Recall@1: {average_recall_1:.1%}")
    print(f"Recall@3: {average_recall_3:.1%}")
    print(f"MRR: {mean_reciprocal_rank:.3f}")


if __name__ == "__main__":
    main()
