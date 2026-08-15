from rag_autopsy.chunking import Chunk
from rag_autopsy.diagnostics import (
    CitationDiagnosisType,
    CitationSupportDiagnosisType,
    DiagnosisType,
    RAGVerdictType,
    run_rag_autopsy,
)
from rag_autopsy.generation import GroundedGenerator
from rag_autopsy.retrieval import SearchResult


class FakeRetriever:
    def __init__(
        self,
        results,
    ) -> None:
        self.results = results
        self.calls = []

    def search(
        self,
        query: str,
        top_k: int = 3,
    ):
        self.calls.append(
            (query, top_k)
        )
        return self.results[:top_k]


class FakeLLM:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response

    def generate(
        self,
        prompt: str,
    ) -> str:
        return self.response


def make_result(
    chunk_id: str,
    text: str,
) -> SearchResult:
    return SearchResult(
        score=0.9,
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc",
            text=text,
            start_word=0,
            end_word=20,
        ),
    )


def test_end_to_end_report_combines_all_stages() -> None:
    retrieval_results = [
        make_result(
            "doc::paragraph-0001",
            (
                "Reopen rates declined over "
                "the next two reporting periods."
            ),
        )
    ]

    retriever = FakeRetriever(
        retrieval_results
    )

    generator = GroundedGenerator(
        llm=FakeLLM(

                "Reopen rates declined "
                "[doc::paragraph-0001]."

        )
    )

    report = run_rag_autopsy(
        question="What was the result?",
        retriever=retriever,
        generator=generator,
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
        top_k=3,
    )

    assert (
        report.retrieval.diagnosis
        == DiagnosisType.SUCCESS
    )

    assert (
        report.citations.diagnosis
        == CitationDiagnosisType.VALID_CITATION
    )

    assert (
        report.citation_support.diagnosis
        == CitationSupportDiagnosisType.SUPPORTED_BY_TEXT
    )

    assert report.generation.answer == (
        "Reopen rates declined "
        "[doc::paragraph-0001]."
    )

    assert retriever.calls == [
        (
            "What was the result?",
            3,
        )
    ]


def test_report_preserves_retrieval_failure() -> None:
    retrieval_results = [
        make_result(
            "doc::paragraph-0000",
            "The incident investigation began.",
        )
    ]

    retriever = FakeRetriever(
        retrieval_results
    )

    generator = GroundedGenerator(
        llm=FakeLLM(

                "The investigation began "
                "[doc::paragraph-0000]."

        )
    )

    report = run_rag_autopsy(
        question="What fixed the issue?",
        retriever=retriever,
        generator=generator,
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
    )

    assert (
        report.retrieval.diagnosis
        == DiagnosisType.RETRIEVAL_MISS
    )

    assert (
        report.citations.diagnosis
        == CitationDiagnosisType.VALID_CITATION
    )


def test_report_retains_retrieved_results() -> None:
    retrieval_results = [
        make_result(
            "doc::paragraph-0001",
            "Reopen rates declined.",
        )
    ]

    report = run_rag_autopsy(
        question="What happened?",
        retriever=FakeRetriever(
            retrieval_results
        ),
        generator=GroundedGenerator(
            llm=FakeLLM(

                    "Reopen rates declined "
                    "[doc::paragraph-0001]."

            )
        ),
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
    )

    assert (
        report.retrieval_results[0]
        .chunk.chunk_id
        == "doc::paragraph-0001"
    )


def test_report_includes_primary_verdict() -> None:
    retrieval_results = [
        make_result(
            "doc::paragraph-0001",
            (
                "Reopen rates declined over "
                "the next two reporting periods."
            ),
        )
    ]

    report = run_rag_autopsy(
        question="What happened?",
        retriever=FakeRetriever(
            retrieval_results
        ),
        generator=GroundedGenerator(
            llm=FakeLLM(

                    "Reopen rates declined "
                    "[doc::paragraph-0001]."

            )
        ),
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
    )

    assert (
        report.verdict.diagnosis
        == RAGVerdictType.SUCCESS
    )
