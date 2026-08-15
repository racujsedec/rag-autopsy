import pytest

import rag_autopsy.retrieval.pgvector as pgvector_module
from rag_autopsy.retrieval import PgVectorRetriever


class FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def encode_query(
        self,
        query,
        convert_to_numpy,
        normalize_embeddings,
    ):
        self.calls.append(
            (
                query,
                convert_to_numpy,
                normalize_embeddings,
            )
        )
        return [0.1, 0.2, 0.3]


class FakeCursor:
    def __init__(self, rows) -> None:
        self.rows = rows
        self.executed_params = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def execute(
        self,
        query,
        params,
    ) -> None:
        self.executed_params = params

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows) -> None:
        self.cursor_instance = FakeCursor(rows)

    def cursor(self):
        return self.cursor_instance


def disable_vector_registration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pgvector_module,
        "register_vector",
        lambda connection: None,
    )


def test_search_maps_database_rows_to_search_results(
    monkeypatch,
) -> None:
    disable_vector_registration(monkeypatch)

    connection = FakeConnection(
        [
            (
                "doc::paragraph-0001",
                "doc",
                "Answer paragraph.",
                10,
                20,
                0.91,
            ),
            (
                "doc::paragraph-0000",
                "doc",
                "Context paragraph.",
                0,
                10,
                0.82,
            ),
        ]
    )

    model = FakeModel()

    retriever = PgVectorRetriever(
        connection=connection,
        model=model,
    )

    results = retriever.search(
        "What happened?",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].score == pytest.approx(
        0.91
    )
    assert (
        results[0].chunk.chunk_id
        == "doc::paragraph-0001"
    )
    assert results[0].chunk.document_id == "doc"
    assert (
        results[0].chunk.text
        == "Answer paragraph."
    )
    assert results[0].chunk.start_word == 10
    assert results[0].chunk.end_word == 20


def test_search_encodes_query_and_passes_top_k(
    monkeypatch,
) -> None:
    disable_vector_registration(monkeypatch)

    connection = FakeConnection([])
    model = FakeModel()

    retriever = PgVectorRetriever(
        connection=connection,
        model=model,
    )

    retriever.search(
        "service desk changes",
        top_k=5,
    )

    assert model.calls == [
        (
            "service desk changes",
            True,
            True,
        )
    ]

    params = (
        connection
        .cursor_instance
        .executed_params
    )

    assert params[-1] == 5


def test_invalid_top_k_is_rejected(
    monkeypatch,
) -> None:
    disable_vector_registration(monkeypatch)

    retriever = PgVectorRetriever(
        connection=FakeConnection([]),
        model=FakeModel(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        retriever.search(
            "query",
            top_k=0,
        )
