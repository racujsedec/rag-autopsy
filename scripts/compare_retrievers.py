import json
from pathlib import Path

from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig
from rag_autopsy.diagnostics import diagnose_retrieval
from rag_autopsy.evaluation import recall_at_k, reciprocal_rank
from rag_autopsy.evaluation.ground_truth import resolve_ground_truth
from rag_autopsy.retrieval import (
    BM25Retriever,
    HybridRetriever,
    RerankingRetriever,
    SemanticRetriever,
)


def load_chunks():
    raw_dir = Path("data/raw")

    chunker = FixedSizeChunker(
        ChunkingConfig(
            chunk_size=45,
            overlap=8,
        )
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
        Path("data/evaluation/questions.json").read_text()
    )


def evaluate_retriever(
    name,
    retriever,
    questions,
    chunks,
):
    recall_1_scores = []
    recall_3_scores = []
    rr_scores = []

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    for item in questions:
        question = item["question"]
        evidence_text = item["evidence_text"]

        ground_truth = resolve_ground_truth(
            chunks=chunks,
            evidence_text=evidence_text,
        )

        relevant = ground_truth.relevant_chunk_ids

        results = retriever.search(
            question,
            top_k=3,
        )

        r1 = recall_at_k(
            results,
            relevant,
            k=1,
        )

        r3 = recall_at_k(
            results,
            relevant,
            k=3,
        )

        rr = reciprocal_rank(
            results,
            relevant,
        )

        diagnosis = diagnose_retrieval(
            results,
            relevant,
        )

        recall_1_scores.append(r1)
        recall_3_scores.append(r3)
        rr_scores.append(rr)

        top_chunk = (
            results[0].chunk.chunk_id
            if results
            else "NO RESULT"
        )

        print(
            f"{item['question_id']} | "
            f"Top: {top_chunk} | "
            f"Coverage: {ground_truth.max_coverage:.1%} | "
            f"Diagnosis: {diagnosis.diagnosis.value}"
        )

    return {
        "recall@1": (
            sum(recall_1_scores) / len(recall_1_scores)
        ),
        "recall@3": (
            sum(recall_3_scores) / len(recall_3_scores)
        ),
        "mrr": (
            sum(rr_scores) / len(rr_scores)
        ),
    }


def main():
    chunks = load_chunks()
    questions = load_questions()

    print("\nLoading BM25...")
    bm25 = BM25Retriever(chunks)

    print("Loading semantic model...")
    semantic = SemanticRetriever(chunks)
    
    print("Loading hybrid retriever...")
    hybrid = HybridRetriever(chunks)

    print("Loading cross-encoder reranker...")
    reranked = RerankingRetriever(
        base_retriever=hybrid,
        candidate_k=6,
    )

    bm25_metrics = evaluate_retriever(
        "BM25",
        bm25,
        questions,
        chunks,
    )

    semantic_metrics = evaluate_retriever(
        "SEMANTIC",
        semantic,
        questions,
        chunks,
    )

    hybrid_metrics = evaluate_retriever(
        "HYBRID RRF",
        hybrid,
        questions,
        chunks,
    )

    reranked_metrics = evaluate_retriever(
        "HYBRID + RERANKER",
        reranked,
        questions,
        chunks,
    )

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(
        f"{'Retriever':<15}"
        f"{'Recall@1':<15}"
        f"{'Recall@3':<15}"
        f"{'MRR':<15}"
    )

    print("-" * 60)

    print(
        f"{'BM25':<15}"
        f"{bm25_metrics['recall@1']:<15.1%}"
        f"{bm25_metrics['recall@3']:<15.1%}"
        f"{bm25_metrics['mrr']:<15.3f}"
    )

    print(
        f"{'Semantic':<15}"
        f"{semantic_metrics['recall@1']:<15.1%}"
        f"{semantic_metrics['recall@3']:<15.1%}"
        f"{semantic_metrics['mrr']:<15.3f}"
    )
   
    print(
        f"{'Hybrid RRF':<15}"
        f"{hybrid_metrics['recall@1']:<15.1%}"
        f"{hybrid_metrics['recall@3']:<15.1%}"
        f"{hybrid_metrics['mrr']:<15.3f}"
    )

    print(
        f"{'Hybrid+Rerank':<15}"
        f"{reranked_metrics['recall@1']:<15.1%}"
        f"{reranked_metrics['recall@3']:<15.1%}"
        f"{reranked_metrics['mrr']:<15.3f}"
    )

if __name__ == "__main__":
    main()
