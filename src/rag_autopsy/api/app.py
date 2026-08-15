import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="RAG Autopsy",
    description=(
        "API for RAG retrieval, generation, "
        "and failure diagnostics."
    ),
    version="0.1.0",
)


class AutopsyRequest(BaseModel):
    question_id: str
    generate: bool = False
    top_k: int = 3


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "rag-autopsy",
    }


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


def run_benchmark_retrieval_data(
    question_id: str,
    top_k: int,
) -> dict[str, object]:
    import psycopg

    from rag_autopsy.diagnostics import (
        diagnose_retrieval,
    )
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

        relevant_chunk_ids = list(
            ground_truth.relevant_chunk_ids
        )
    else:
        relevant_chunk_ids = []

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
        diagnosis = diagnose_retrieval(
            results=results,
            relevant_chunk_ids=(
                relevant_chunk_ids
            ),
        ).diagnosis.value
    else:
        diagnosis = "NOT_APPLICABLE"

    return {
        "question_id": question_id,
        "question": item["question"],
        "diagnosis": diagnosis,
        "relevant_chunk_ids": (
            relevant_chunk_ids
        ),
        "retrieved_chunks": [
            {
                "rank": rank,
                "chunk_id": (
                    result.chunk.chunk_id
                ),
                "score": result.score,
            }
            for rank, result in enumerate(
                results,
                start=1,
            )
        ],
    }


@app.post("/autopsy")
def autopsy(
    request: AutopsyRequest,
) -> dict[str, object]:
    if request.generate:
        raise HTTPException(
            status_code=501,
            detail=(
                "Generation through the API "
                "is not implemented yet."
            ),
        )

    try:
        return run_benchmark_retrieval_data(
            question_id=request.question_id,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
