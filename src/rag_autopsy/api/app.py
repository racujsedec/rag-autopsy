from fastapi import FastAPI

app = FastAPI(
    title="RAG Autopsy",
    description=(
        "API for RAG retrieval, generation, "
        "and failure diagnostics."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "rag-autopsy",
    }
