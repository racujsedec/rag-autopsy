from rag_autopsy.generation.openai_llm import OpenAIResponsesLLM


class FakeResponse:
    output_text = "Grounded answer [doc::paragraph-0001]"


class FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


def test_openai_responses_llm_generates_text():
    client = FakeClient()

    llm = OpenAIResponsesLLM(
        client=client,
        model="test-model",
    )

    answer = llm.generate(
        "Answer using retrieved evidence."
    )

    assert answer == (
        "Grounded answer "
        "[doc::paragraph-0001]"
    )

    assert client.responses.calls == [
        {
            "model": "test-model",
            "input": (
                "Answer using retrieved evidence."
            ),
        }
    ]
