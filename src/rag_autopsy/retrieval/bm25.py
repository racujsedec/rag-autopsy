import math
import re
from collections import Counter
from dataclasses import dataclass

from rag_autopsy.chunking import Chunk


def tokenize(text: str) -> list[str]:
    """Convert text into simple lowercase word tokens."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


@dataclass(frozen=True)
class SearchResult:
    score: float
    chunk: Chunk


class BM25Retriever:
    """Small BM25 implementation for our retrieval baseline."""

    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.tokenized_documents = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )

        self.document_frequencies = self._calculate_document_frequencies()

    def _calculate_document_frequencies(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()

        for tokens in self.tokenized_documents:
            unique_terms = set(tokens)

            for term in unique_terms:
                frequencies[term] += 1

        return frequencies

    def _idf(self, term: str) -> float:
        number_of_documents = len(self.chunks)
        document_frequency = self.document_frequencies.get(term, 0)

        if document_frequency == 0:
            return 0.0

        return math.log(
            1
            + (
                number_of_documents
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[SearchResult]:

        query_terms = tokenize(query)

        results: list[SearchResult] = []

        for chunk, document_tokens, document_length in zip(
            self.chunks,
            self.tokenized_documents,
            self.document_lengths,
        ):
            term_counts = Counter(document_tokens)

            score = 0.0

            for term in query_terms:
                term_frequency = term_counts.get(term, 0)

                if term_frequency == 0:
                    continue

                idf = self._idf(term)

                length_normalization = (
                    1
                    - self.b
                    + self.b
                    * (
                        document_length
                        / self.average_document_length
                    )
                )

                numerator = (
                    term_frequency
                    * (self.k1 + 1)
                )

                denominator = (
                    term_frequency
                    + self.k1
                    * length_normalization
                )

                score += (
                    idf
                    * numerator
                    / denominator
                )

            if score > 0:
                results.append(
                    SearchResult(
                        score=score,
                        chunk=chunk,
                    )
                )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]
