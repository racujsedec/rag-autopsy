from rag_autopsy.chunking import Chunk
from rag_autopsy.diagnostics import (
    format_rag_autopsy_report,
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

    def search(
        self,
        query: str,
        top_k: int = 3,
    ):
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
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        score=score,
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc",
            text=text,
            start_word=0,
            end_word=20,
        ),
    )


def build_success_report():
    return run_rag_autopsy(
        question="What happened?",
        retriever=FakeRetriever(
            [
                make_result(
                    "doc::paragraph-0001",
                    (
                        "Reopen rates declined over "
                        "the next two reporting periods."
                    ),
                )
            ]
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


def test_formatter_includes_question_and_verdict() -> None:
    formatted = format_rag_autopsy_report(
        build_success_report()
    )

    assert "RAG AUTOPSY REPORT" in formatted
    assert "What happened?" in formatted
    assert "PRIMARY DIAGNOSIS: SUCCESS" in formatted


def test_formatter_includes_stage_diagnoses() -> None:
    formatted = format_rag_autopsy_report(
        build_success_report()
    )

    assert "Retrieval: SUCCESS" in formatted
    assert "Citation validity: VALID_CITATION" in formatted
    assert (
        "Citation support: SUPPORTED_BY_TEXT"
        in formatted
    )


def test_formatter_includes_generated_answer() -> None:
    formatted = format_rag_autopsy_report(
        build_success_report()
    )

    assert (
        "Reopen rates declined "
        "[doc::paragraph-0001]."
        in formatted
    )


def test_formatter_includes_ranked_retrieved_chunks() -> None:
    formatted = format_rag_autopsy_report(
        build_success_report()
    )

    assert (
        "1. doc::paragraph-0001"
        in formatted
    )
    assert "score=0.9000" in formatted


def test_formatter_includes_citation_coverage() -> None:
    formatted = format_rag_autopsy_report(
        build_success_report()
    )

    assert (
        "Citation coverage: COMPLETE_CITATION_COVERAGE"
        in formatted
    )
    assert "Coverage: 100.0%" in formatted
