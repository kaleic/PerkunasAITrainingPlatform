from __future__ import annotations

from fastapi.testclient import TestClient

from kvserve.app import create_app


def test_chat_completion_and_aliases() -> None:
    client = TestClient(create_app())
    headers = {"Authorization": "Bearer dev-token"}
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "dev/kv-echo-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["usage"]["total_tokens"] > 0

    rerank = client.post(
        "/v1/reranking",
        headers=headers,
        json={
            "model": "dev/kv-rerank",
            "query": "kv memory",
            "documents": ["memory cache", "banana"],
        },
    )
    assert rerank.status_code == 200
    assert rerank.json()["results"][0]["index"] == 0
