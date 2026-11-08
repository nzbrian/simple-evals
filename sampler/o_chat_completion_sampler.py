import time
from typing import Any

import openai
from openai import OpenAI

from ..eval_types import MessageList, SamplerBase, SamplerResponse


class OChatCompletionSampler(SamplerBase):
    """
    Sample from an OpenAI chat completions API for o series (reasoning)
    models, or any OpenAI-compatible endpoint that serves them.
    """

    def __init__(
        self,
        *,
        reasoning_effort: str | None = None,
        model: str = "o1-mini",
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """
        Args:
            base_url: Base URL of the endpoint, e.g. "http://localhost:8000/v1".
                Defaults to the OPENAI_BASE_URL environment variable, then to
                the official OpenAI API.
            api_key: API key for the endpoint. Defaults to the OPENAI_API_KEY
                environment variable. Pass any non-empty string (e.g. "dummy")
                for local servers that do not require authentication.
        """
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.image_format = "url"
        self.reasoning_effort = reasoning_effort

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
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,  # type: ignore[arg-type]
                    reasoning_effort=self.reasoning_effort,  # type: ignore[arg-type]
                )
                content = response.choices[0].message.content
                return SamplerResponse(
                    response_text=content or "",
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="",
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
