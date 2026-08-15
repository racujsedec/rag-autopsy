from rag_autopsy.chunking import Chunk
from rag_autopsy.diagnostics import (
    RAGVerdictType,
    diagnose_rag_verdict,
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


def build_report(
    retrieval_results,
    answer,
    relevant_chunk_ids,
):
    return run_rag_autopsy(
        question="What happened?",
        retriever=FakeRetriever(
            retrieval_results
        ),
        generator=GroundedGenerator(
            llm=FakeLLM(answer)
        ),
        relevant_chunk_ids=(
            relevant_chunk_ids
        ),
    )


def test_retrieval_miss_has_highest_priority() -> None:
    report = build_report(
        retrieval_results=[
            make_result(
                "doc::paragraph-0000",
                "Investigation started.",
            )
        ],
        answer=(
            "Unsupported claim "
            "[doc::paragraph-9999]."
        ),
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
    )

    verdict = diagnose_rag_verdict(
        report
    )

    assert (
        verdict.diagnosis
        == RAGVerdictType.RETRIEVAL_MISS
    )


def test_ranking_failure_is_reported() -> None:
    report = build_report(
        retrieval_results=[
            make_result(
                "doc::paragraph-0000",
                "Background context.",
            ),
            make_result(
                "doc::paragraph-0001",
                "Reopen rates declined.",
            ),
        ],
        answer=(
            "Reopen rates declined "
            "[doc::paragraph-0001]."
        ),
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
    )

    verdict = diagnose_rag_verdict(
        report
    )

    assert (
        verdict.diagnosis
        == RAGVerdictType.RANKING_FAILURE
    )


def test_invalid_citation_is_reported() -> None:
    report = build_report(
        retrieval_results=[
            make_result(
                "doc::paragraph-0001",
                "Reopen rates declined.",
            )
        ],
        answer=(
            "Reopen rates declined "
            "[doc::paragraph-9999]."
        ),
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
    )

    verdict = diagnose_rag_verdict(
        report
    )

    assert (
        verdict.diagnosis
        == RAGVerdictType.INVALID_CITATION
    )


def test_low_textual_support_is_reported() -> None:
    report = build_report(
        retrieval_results=[
            make_result(
                "doc::paragraph-0001",
                "Reopen rates declined.",
            )
        ],
        answer=(
            "Revenue doubled dramatically "
            "[doc::paragraph-0001]."
        ),
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
    )

    verdict = diagnose_rag_verdict(
        report
    )

    assert (
        verdict.diagnosis
        == RAGVerdictType.LOW_TEXTUAL_SUPPORT
    )


def test_missing_citation_is_reported() -> None:
    report = build_report(
        retrieval_results=[
            make_result(
                "doc::paragraph-0001",
                "Reopen rates declined.",
            )
        ],
        answer="Reopen rates declined.",
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
    )

    verdict = diagnose_rag_verdict(
        report
    )

    assert (
        verdict.diagnosis
        == RAGVerdictType.NO_CITATION
    )


def test_clean_pipeline_is_success() -> None:
    report = build_report(
        retrieval_results=[
            make_result(
                "doc::paragraph-0001",
                (
                    "Reopen rates declined over "
                    "the next two reporting periods."
                ),
            )
        ],
        answer=(
            "Reopen rates declined "
            "[doc::paragraph-0001]."
        ),
        relevant_chunk_ids=[
            "doc::paragraph-0001",
        ],
    )

    verdict = diagnose_rag_verdict(
        report
    )

    assert (
        verdict.diagnosis
        == RAGVerdictType.SUCCESS
    )
