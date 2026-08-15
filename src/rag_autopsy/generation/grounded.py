import re
from dataclasses import dataclass

from rag_autopsy.retrieval import SearchResult


@dataclass(frozen=True)
class GroundedGenerationResult:
    answer: str
    cited_chunk_ids: tuple[str, ...]
    invalid_citation_ids: tuple[str, ...]


class GroundedGenerator:
    """Generate an answer grounded in retrieved canonical chunks."""

    def __init__(
        self,
        llm,
    ) -> None:
        self.llm = llm

    def generate(
        self,
        question: str,
        results: list[SearchResult],
    ) -> GroundedGenerationResult:
        prompt = self._build_prompt(
            question=question,
            results=results,
        )

        answer = self.llm.generate(
            prompt
        )

        valid_chunk_ids = {
            result.chunk.chunk_id
            for result in results
        }

        citation_ids = self._extract_citations(
            answer
        )

        valid_citations = tuple(
            citation_id
            for citation_id in citation_ids
            if citation_id in valid_chunk_ids
        )

        invalid_citations = tuple(
            citation_id
            for citation_id in citation_ids
            if citation_id not in valid_chunk_ids
        )

        return GroundedGenerationResult(
            answer=answer,
            cited_chunk_ids=valid_citations,
            invalid_citation_ids=invalid_citations,
        )

    def _build_prompt(
        self,
        question: str,
        results: list[SearchResult],
    ) -> str:
        context_blocks = []

        for result in results:
            context_blocks.append(

                    f"[{result.chunk.chunk_id}]\n"
                    f"{result.chunk.text}"

            )

        context = "\n\n".join(
            context_blocks
        )

        return (
            "Answer the question using only the "
            "retrieved context below.\n"
            "Cite supporting chunks using their exact "
            "chunk IDs in square brackets.\n"
            "If the context does not support an answer, "
            "say that the answer is not available in the "
            "retrieved context.\n\n"
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{context}"
        )

    @staticmethod
    def _extract_citations(
        answer: str,
    ) -> tuple[str, ...]:
        matches = re.findall(
            r"\[([^\[\]]+)\]",
            answer,
        )

        return tuple(
            dict.fromkeys(matches)
        )
