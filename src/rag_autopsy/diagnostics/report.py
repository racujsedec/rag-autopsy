from dataclasses import dataclass

from rag_autopsy.generation import (
    GroundedGenerationResult,
    GroundedGenerator,
)
from rag_autopsy.retrieval import SearchResult

from .autopsy import (
    AutopsyResult,
    diagnose_retrieval,
)
from .citation import (
    CitationAutopsyResult,
    diagnose_citations,
)
from .citation_coverage import (
    CitationCoverageAutopsyResult,
    diagnose_citation_coverage,
)
from .citation_support import (
    CitationSupportAutopsyResult,
    diagnose_citation_support,
)
from .verdict import (
    RAGVerdictResult,
    diagnose_rag_stages,
)


@dataclass(frozen=True)
class RAGAutopsyReport:
    question: str
    retrieval_results: tuple[SearchResult, ...]
    retrieval: AutopsyResult
    generation: GroundedGenerationResult
    citations: CitationAutopsyResult
    citation_support: CitationSupportAutopsyResult
    citation_coverage: CitationCoverageAutopsyResult
    verdict: RAGVerdictResult


def run_rag_autopsy(
    question: str,
    retriever,
    generator: GroundedGenerator,
    relevant_chunk_ids: list[str],
    top_k: int = 3,
) -> RAGAutopsyReport:
    retrieval_results = retriever.search(
        question,
        top_k=top_k,
    )

    retrieval_diagnosis = diagnose_retrieval(
        retrieval_results,
        relevant_chunk_ids,
    )

    generation_result = generator.generate(
        question=question,
        results=retrieval_results,
    )

    citation_diagnosis = diagnose_citations(
        generation_result
    )

    citation_support_diagnosis = (
        diagnose_citation_support(
            generation_result=generation_result,
            retrieval_results=retrieval_results,
        )
    )

    citation_coverage_diagnosis = (
        diagnose_citation_coverage(
            generation_result
        )
    )

    verdict = diagnose_rag_stages(
        retrieval=retrieval_diagnosis,
        citations=citation_diagnosis,
        citation_support=citation_support_diagnosis,
        citation_coverage=citation_coverage_diagnosis,
    )

    return RAGAutopsyReport(
        question=question,
        retrieval_results=tuple(
            retrieval_results
        ),
        retrieval=retrieval_diagnosis,
        generation=generation_result,
        citations=citation_diagnosis,
        citation_support=(
            citation_support_diagnosis
        ),
        citation_coverage=(
            citation_coverage_diagnosis
        ),
        verdict=verdict,
    )
