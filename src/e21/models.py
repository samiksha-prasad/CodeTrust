"""Models under test, addressed as "provider:model".

anthropic:claude-opus-5-5     Anthropic SDK
openai:<model>                OpenAI SDK
compat:<model>                any OpenAI-compatible endpoint (Gemini, Groq, Ollama, OpenRouter);
                              set E21_COMPAT_BASE_URL and E21_COMPAT_API_KEY
mock:fabricator               offline stand-in that always invents a meaning
grounded+<any of the above>   the same model behind the lookup layer in grounding.py
"""

import os

REFUSAL = "[REFUSAL]"


class AnthropicModel:
    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def ask(self, prompt: str, system: str | None = None) -> str:
        kwargs = {"system": system} if system else {}
        # No server-side refusal fallback here: it would answer with a different model than
        # the one being measured.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            return REFUSAL
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIModel:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, label: str = "openai"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.name = f"{label}:{model}"

    def ask(self, prompt: str, system: str | None = None) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content or ""


class MockModel:
    """Pipeline testing without API keys. Fabricates unless the grounding layer says not to."""

    def __init__(self, model: str):
        self.name = f"mock:{model}"

    def ask(self, prompt: str, system: str | None = None) -> str:
        if system and "official documentation" in system:
            return system
        if system:
            return "I can't find that code in the official list. Which brand and model is it?"
        return "That code means the drain pump is blocked. Open the back panel and clear it."


def get_model(spec: str, grounder=None):
    if spec.startswith("grounded+"):
        if grounder is None:
            raise ValueError("a grounded+ model needs --codes (and --coverage) to build its lookup table")
        from .grounding import GroundedModel

        return GroundedModel(get_model(spec.removeprefix("grounded+")), grounder)
    provider, _, model = spec.partition(":")
    if provider == "anthropic":
        return AnthropicModel(model)
    if provider == "openai":
        return OpenAIModel(model)
    if provider == "compat":
        return OpenAIModel(
            model,
            base_url=os.environ["E21_COMPAT_BASE_URL"],
            api_key=os.environ.get("E21_COMPAT_API_KEY", "none"),
            label="compat",
        )
    if provider == "mock":
        return MockModel(model)
    raise ValueError(f"unknown provider in model spec {spec!r}")
