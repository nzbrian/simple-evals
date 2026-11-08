import time
from typing import Any

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from an OpenAI chat completions API, or any OpenAI-compatible
    endpoint (e.g. vLLM, Ollama, LM Studio, OpenRouter, Together, Groq).
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_body: dict | None = None,
    ):
        """
        Args:
            base_url: Base URL of the endpoint, e.g. "http://localhost:8000/v1".
                Defaults to the OPENAI_BASE_URL environment variable, then to
                the official OpenAI API.
            api_key: API key for the endpoint. Defaults to the OPENAI_API_KEY
                environment variable. Pass any non-empty string (e.g. "dummy")
                for local servers that do not require authentication.
            extra_body: Extra keys merged into the request body. Used for
                endpoint-specific options, e.g. disabling thinking on reasoning
                models: {"chat_template_kwargs": {"enable_thinking": False}}.
        """
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,  # type: ignore[arg-type]
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    extra_body=self.extra_body,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
