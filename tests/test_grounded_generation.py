from rag_autopsy.chunking import Chunk
from rag_autopsy.generation import GroundedGenerator
from rag_autopsy.retrieval import SearchResult


class FakeLLM:
    def __init__(
        self,
        response: str,
    ) -> None:
        self.response = response
        self.prompts = []

    def generate(
        self,
        prompt: str,
    ) -> str:
        self.prompts.append(prompt)
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
            end_word=10,
        ),
    )


def test_prompt_contains_question_and_canonical_chunks() -> None:
    llm = FakeLLM(
        "The change reduced reopen rates."
    )

    generator = GroundedGenerator(
        llm=llm
    )

    generator.generate(
        question="What changed?",
        results=[
            make_result(
                "doc::paragraph-0001",
                "Resolution codes became mandatory.",
            )
        ],
    )

    prompt = llm.prompts[0]

    assert "What changed?" in prompt
    assert "doc::paragraph-0001" in prompt
    assert (
        "Resolution codes became mandatory."
        in prompt
    )


def test_generated_answer_is_returned() -> None:
    llm = FakeLLM(
        "Resolution codes became mandatory "
        "[doc::paragraph-0001]."
    )

    generator = GroundedGenerator(
        llm=llm
    )

    result = generator.generate(
        question="What changed?",
        results=[
            make_result(
                "doc::paragraph-0001",
                "Resolution codes became mandatory.",
            )
        ],
    )

    assert result.answer == (
        "Resolution codes became mandatory "
        "[doc::paragraph-0001]."
    )


def test_valid_chunk_citations_are_extracted() -> None:
    llm = FakeLLM(
        "The change reduced reopen rates "
        "[doc::paragraph-0001]."
    )

    generator = GroundedGenerator(
        llm=llm
    )

    result = generator.generate(
        question="What was the result?",
        results=[
            make_result(
                "doc::paragraph-0001",
                "Reopen rates declined.",
            )
        ],
    )

    assert result.cited_chunk_ids == (
        "doc::paragraph-0001",
    )


def test_unknown_citations_are_reported_separately() -> None:
    llm = FakeLLM(
        "The result improved "
        "[doc::paragraph-9999]."
    )

    generator = GroundedGenerator(
        llm=llm
    )

    result = generator.generate(
        question="What happened?",
        results=[
            make_result(
                "doc::paragraph-0001",
                "Reopen rates declined.",
            )
        ],
    )

    assert result.cited_chunk_ids == ()
    assert result.invalid_citation_ids == (
        "doc::paragraph-9999",
    )
