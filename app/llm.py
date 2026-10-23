"""Model objects for the cascade.

A model string is normally handed to Pydantic AI as-is (``openai:gpt-5-mini``,
``mistral:mistral-small-2603``). ``ETCHMEM_RELAX_CHAT_METADATA`` builds every
``openai:`` model with one shared HTTP client that tolerates OpenAI-compatible
gateways. Those gateways sometimes attach a chat ``metadata`` object whose
values are not strings; the OpenAI SDK types that field as a string map and
rejects the whole reply. A metadata object that is already a string map is
left as it is, so a real OpenAI response is unchanged. Other providers are
never wrapped.
"""
from __future__ import annotations

import json
import logging
import os

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def _string_map(value: object) -> bool:
    return isinstance(value, dict) and all(isinstance(item, str) for item in value.values())


class RelaxChatMetadata(httpx.AsyncBaseTransport):
    """Drop a chat-completion ``metadata`` object the OpenAI SDK would reject."""

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if "application/json" not in response.headers.get("content-type", ""):
            return response
        body = b"".join([chunk async for chunk in response.stream])
        await response.aclose()
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if isinstance(data, dict) and "metadata" in data and not _string_map(data["metadata"]):
            data.pop("metadata")
            body = json.dumps(data).encode("utf-8")
        headers = [(key, value) for key, value in response.headers.multi_items()
                   if key.lower() not in ("content-length", "transfer-encoding", "content-encoding")]
        return httpx.Response(response.status_code, headers=headers, content=body, request=request,
                              extensions=response.extensions)


def model_for(spec: str):
    """The argument for ``Agent(...)``: the string itself, or a relaxed OpenAI model."""
    spec = (spec or "").strip()
    provider, sep, name = spec.partition(":")
    if not (settings.relax_chat_metadata and sep and provider == "openai" and name):
        return spec
    from pydantic_ai.models import get_user_agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    # Same timeouts and User-Agent as Pydantic AI's own client. Identity encoding
    # because this transport rewrites the JSON body. base_url None keeps
    # OPENAI_BASE_URL (or api.openai.com). A missing key must not stop startup.
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        logger.error("OPENAI_API_KEY is not set: openai models will refuse every call.")
        key = "missing-OPENAI_API_KEY"
    client = httpx.AsyncClient(
        transport=RelaxChatMetadata(),
        timeout=httpx.Timeout(600.0, connect=5.0),
        headers={"User-Agent": get_user_agent(), "Accept-Encoding": "identity"},
    )
    return OpenAIChatModel(name, provider=OpenAIProvider(api_key=key, http_client=client))
