"""OpenAI-compatible gateways: one switch relaxes chat metadata for every model."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import settings
from app.llm import RelaxChatMetadata, model_for

_KIMCHI = {
    "id": "c1", "object": "chat.completion", "created": 1, "model": "kimi-k3",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "Hello there"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    "metadata": {"weight_version": "default",
                 "weight_versions": [{"version": "default", "start": 0, "end": 160}]},
}


def _client(payload: dict, transport_cls=RelaxChatMetadata) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, json=payload)

    return httpx.AsyncClient(transport=transport_cls(httpx.MockTransport(handler)))


def test_a_string_metadata_map_is_left_alone():
    payload = {**_KIMCHI, "metadata": {"source": "lift"}}

    async def read() -> dict:
        response = await _client(payload).post("https://gw.example/v1/chat/completions")
        return response.json()

    assert asyncio.run(read())["metadata"] == {"source": "lift"}


def test_non_string_metadata_is_removed():
    async def read() -> dict:
        response = await _client(_KIMCHI).post("https://gw.example/v1/chat/completions")
        return response.json()

    body = asyncio.run(read())
    assert "metadata" not in body and body["choices"][0]["message"]["content"] == "Hello there"


def test_a_stream_passes_through():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"data: hi\n\n")

    client = httpx.AsyncClient(transport=RelaxChatMetadata(httpx.MockTransport(handler)))
    response = asyncio.run(client.post("https://gw.example/v1/chat/completions"))
    assert response.text == "data: hi\n\n"


def test_the_switch_is_off_by_default_and_other_providers_stay_strings(monkeypatch):
    monkeypatch.setattr(settings, "relax_chat_metadata", False)
    assert model_for("openai:gpt-5-mini") == "openai:gpt-5-mini"
    monkeypatch.setattr(settings, "relax_chat_metadata", True)
    assert model_for("mistral:mistral-small-2603") == "mistral:mistral-small-2603"
    assert model_for("") == ""


def test_with_the_switch_an_openai_model_reads_a_gateway_reply(monkeypatch):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    monkeypatch.setattr(settings, "relax_chat_metadata", True)
    built = model_for("openai:kimi-k3")
    assert isinstance(built, OpenAIChatModel) and built.model_name == "kimi-k3"

    model = OpenAIChatModel("kimi-k3", provider=OpenAIProvider(
        base_url="https://gw.example/v1", api_key="k", http_client=_client(_KIMCHI)))
    assert asyncio.run(Agent(model).run("hi")).output == "Hello there"
