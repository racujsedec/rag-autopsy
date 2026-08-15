import pytest

import rag_autopsy.indexing.pgvector as pgvector_module
from rag_autopsy.chunking import Chunk
from rag_autopsy.indexing import PgVectorIndexer


class FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def encode_document(
        self,
        texts,
        convert_to_numpy,
        normalize_embeddings,
    ):
        self.calls.append(
            (
                texts,
                convert_to_numpy,
                normalize_embeddings,
            )
        )

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]


class FakeCursor:
    def __init__(self) -> None:
        self.rows = None

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        return False

    def executemany(
        self,
        query,
        rows,
    ) -> None:
        self.rows = list(rows)


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()
        self.commit_count = 0

    def cursor(self):
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1


def make_chunk(
    chunk_id: str,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        text=text,
        start_word=0,
        end_word=5,
    )


def disable_vector_registration(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pgvector_module,
        "register_vector",
        lambda connection: None,
    )


def test_upsert_chunks_encodes_documents(
    monkeypatch,
) -> None:
    disable_vector_registration(monkeypatch)

    model = FakeModel()
    connection = FakeConnection()

    indexer = PgVectorIndexer(
        connection=connection,
        model=model,
    )

    indexer.upsert_chunks(
        [
            make_chunk(
                "doc::paragraph-0000",
                "First paragraph.",
            ),
            make_chunk(
                "doc::paragraph-0001",
                "Second paragraph.",
            ),
        ]
    )

    assert model.calls == [
        (
            [
                "First paragraph.",
                "Second paragraph.",
            ],
            True,
            True,
        )
    ]


def test_upsert_chunks_writes_expected_rows(
    monkeypatch,
) -> None:
    disable_vector_registration(monkeypatch)

    connection = FakeConnection()

    indexer = PgVectorIndexer(
        connection=connection,
        model=FakeModel(),
    )

    chunks = [
        make_chunk(
            "doc::paragraph-0000",
            "First paragraph.",
        )
    ]

    count = indexer.upsert_chunks(chunks)

    assert count == 1
    assert connection.commit_count == 1

    rows = connection.cursor_instance.rows

    assert rows is not None
    assert rows[0][0] == "doc::paragraph-0000"
    assert rows[0][1] == "doc"
    assert rows[0][2] == "First paragraph."
    assert rows[0][3] == "First paragraph."
    assert rows[0][4] == 0
    assert rows[0][5] == 5


def test_empty_chunk_list_does_nothing(
    monkeypatch,
) -> None:
    disable_vector_registration(monkeypatch)

    model = FakeModel()
    connection = FakeConnection()

    indexer = PgVectorIndexer(
        connection=connection,
        model=model,
    )

    count = indexer.upsert_chunks([])

    assert count == 0
    assert model.calls == []
    assert connection.cursor_instance.rows is None
    assert connection.commit_count == 0


def test_separate_retrieval_chunks_are_used_for_embeddings(
    monkeypatch,
) -> None:
    disable_vector_registration(monkeypatch)

    model = FakeModel()
    connection = FakeConnection()

    canonical = [
        make_chunk(
            "doc::paragraph-0001",
            "Canonical answer paragraph.",
        )
    ]

    retrieval = [
        make_chunk(
            "doc::paragraph-0001",
            "Previous context. Canonical answer paragraph.",
        )
    ]

    indexer = PgVectorIndexer(
        connection=connection,
        model=model,
    )

    indexer.upsert_chunks(
        canonical,
        retrieval_chunks=retrieval,
    )

    assert model.calls == [
        (
            [
                "Previous context. Canonical answer paragraph.",
            ],
            True,
            True,
        )
    ]

    row = connection.cursor_instance.rows[0]

    assert row[2] == "Canonical answer paragraph."
    assert row[3] == (
        "Previous context. Canonical answer paragraph."
    )


def test_retrieval_chunk_ids_must_match_canonical_chunks(
    monkeypatch,
) -> None:
    disable_vector_registration(monkeypatch)

    indexer = PgVectorIndexer(
        connection=FakeConnection(),
        model=FakeModel(),
    )

    canonical = [
        make_chunk(
            "doc::paragraph-0000",
            "Canonical text.",
        )
    ]

    retrieval = [
        make_chunk(
            "doc::paragraph-9999",
            "Wrong retrieval chunk.",
        )
    ]

    with pytest.raises(
        ValueError,
        match="retrieval chunk ids must match canonical chunk ids",
    ):
        indexer.upsert_chunks(
            canonical,
            retrieval_chunks=retrieval,
        )
