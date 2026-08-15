class OpenAIResponsesLLM:
    """Thin adapter around the OpenAI Responses API."""

    def __init__(
        self,
        model: str,
        client=None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()

        self.client = client
        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text
