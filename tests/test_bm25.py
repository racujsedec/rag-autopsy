from rag_autopsy.chunking import FixedSizeChunker
from rag_autopsy.config import ChunkingConfig
from rag_autopsy.retrieval.bm25 import BM25Retriever


def build_test_retriever() -> BM25Retriever:
    documents = {
        "finance": (
            "Arcadia Components reported lower operating margin. "
            "The margin decline was caused by higher logistics expense "
            "and supplier surcharges."
        ),
        "retail": (
            "The retail team improved inventory accuracy and reduced "
            "order cancellations."
        ),
        "platform": (
            "A schema change increased shuffle volume and caused "
            "analytics jobs to run slowly."
        ),
    }

    chunker = FixedSizeChunker(
        ChunkingConfig(
            chunk_size=50,
            overlap=5,
        )
    )

    chunks = []

    for document_id, text in documents.items():
        chunks.extend(
            chunker.chunk(
                document_id=document_id,
                text=text,
            )
        )

    return BM25Retriever(chunks)


def test_financial_query_returns_financial_document_first() -> None:
    retriever = build_test_retriever()

    results = retriever.search(
        "Why did Arcadia operating margin decline?",
        top_k=3,
    )

    assert results
    assert results[0].chunk.document_id == "finance"


def test_unrelated_query_returns_no_results() -> None:
    retriever = build_test_retriever()

    results = retriever.search(
        "volcano eruption magma",
        top_k=3,
    )

    assert results == []


def test_results_are_sorted_by_score() -> None:
    retriever = build_test_retriever()

    results = retriever.search(
        "margin logistics supplier",
        top_k=3,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)
