import importlib

from fastapi.testclient import TestClient

from rag_autopsy.api import app

client = TestClient(app)
api_module = importlib.import_module("rag_autopsy.api.app")


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "rag-autopsy",
    }


def test_autopsy_endpoint_accepts_request(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        api_module,
        "run_benchmark_retrieval_data",
        lambda question_id, top_k: {
            "question_id": question_id,
            "question": "What happened?",
            "diagnosis": "SUCCESS",
            "relevant_chunk_ids": [],
            "retrieved_chunks": [],
        },
    )

    response = client.post(
        "/autopsy",
        json={
            "question_id": "q031",
            "generate": False,
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    assert response.json()["question_id"] == "q031"
    assert response.json()["diagnosis"] == "SUCCESS"


def test_autopsy_endpoint_requires_question_id() -> None:
    response = client.post(
        "/autopsy",
        json={
            "generate": False,
            "top_k": 3,
        },
    )

    assert response.status_code == 422


def test_autopsy_endpoint_returns_retrieval_result(
    monkeypatch,
) -> None:
    def fake_run_retrieval(
        question_id: str,
        top_k: int,
    ) -> dict:
        assert question_id == "q031"
        assert top_k == 3

        return {
            "question_id": "q031",
            "question": "What happened?",
            "diagnosis": "RANKING_FAILURE",
            "relevant_chunk_ids": [
                "doc::paragraph-0001",
            ],
            "retrieved_chunks": [
                {
                    "rank": 1,
                    "chunk_id": "doc::paragraph-0000",
                    "score": 0.9,
                },
                {
                    "rank": 2,
                    "chunk_id": "doc::paragraph-0001",
                    "score": 0.8,
                },
            ],
        }

    monkeypatch.setattr(
        api_module,
        "run_benchmark_retrieval_data",
        fake_run_retrieval,
    )

    response = client.post(
        "/autopsy",
        json={
            "question_id": "q031",
            "generate": False,
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["question_id"] == "q031"
    assert body["diagnosis"] == "RANKING_FAILURE"
    assert body["retrieved_chunks"][1]["chunk_id"] == (
        "doc::paragraph-0001"
    )
