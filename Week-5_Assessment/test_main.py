import pytest
from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace
from fastapi.testclient import TestClient
from main import (
    app,
    get_llm_client,
    get_store,
    get_settings,
    Settings
)
client = TestClient(app)
app_store = {}
mock_llm = Mock()
async def fake_create(**kwargs):
    if kwargs.get("stream"):
        async def generate():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Hello ")
                    )
                ],
                usage=None
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="World")
                    )
                ],
                usage=None
            )
            yield SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5
                )
            )
        return generate()
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello from mock"
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5
        )
    )
mock_llm.chat.completions.create = AsyncMock(
    side_effect=fake_create
)
def override_settings():
    settings = Settings()
    settings.api_key = "mysecret123"
    return settings

app.dependency_overrides[get_llm_client] = (
    lambda: mock_llm
)
app.dependency_overrides[get_store] = (
    lambda: app_store
)
app.dependency_overrides[get_settings] = (
    override_settings
)
@pytest.fixture(autouse=True)
def reset_store():
    app_store.clear()

    mock_llm.chat.completions.create.side_effect = (
        fake_create
    )
    yield
def headers(key="mysecret123"):
    return {
        "x-api-key": key
    }
def create_session():
    response = client.post(
        "/sessions",
        headers=headers()
    )
    return response.json()["session_id"]
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
def test_missing_key():
    response = client.post("/sessions")
    assert response.status_code == 401
def test_wrong_key():
    response = client.post(
        "/sessions",
        headers=headers("wrong")
    )
    assert response.status_code == 403
def test_create_session():
    response = client.post(
        "/sessions",
        headers=headers()
    )
    assert response.status_code == 200
    assert "session_id" in response.json()
def test_empty_message():
    session = create_session()
    response = client.post(
        f"/sessions/{session}/chat",
        headers=headers(),
        json={"message": ""}
    )
    assert response.status_code == 422
def test_long_message():
    session = create_session()
    response = client.post(
        f"/sessions/{session}/chat",
        headers=headers(),
        json={"message": "A" * 2001}
    )
    assert response.status_code == 413
def test_unknown_session():
    response = client.get(
        "/sessions/unknown/messages",
        headers=headers()
    )
    assert response.status_code == 404
def test_chat():
    session = create_session()
    response = client.post(
        f"/sessions/{session}/chat",
        headers=headers(),
        json={"message": "Hello"}
    )
    assert response.status_code == 200
    assert (
        response.json()["answer"]
        == "Hello from mock"
    )
def test_get_messages():
    session = create_session()
    response = client.get(
        f"/sessions/{session}/messages",
        headers=headers()
    )
    assert response.status_code == 200
    assert "messages" in response.json()
def test_usage():
    response = client.get(
        "/usage",
        headers=headers()
    )
    assert response.status_code == 200
    assert "total_tokens" in response.json()
def test_llm_failure():
    session = create_session()
    mock_llm.chat.completions.create.side_effect = (
        Exception("LLM error")
    )
    response = client.post(
        f"/sessions/{session}/chat",
        headers=headers(),
        json={"message": "Hello"}
    )
    assert response.status_code == 502
    assert len(app_store[session].messages) == 1
def test_streaming():
    response = client.post(
        "/chat/stream",
        headers=headers(),
        json={"message": "Hello"}
    )
    assert response.status_code == 200
    assert (
        "text/event-stream"
        in response.headers["content-type"]
    )
    assert "Hello" in response.text
    assert "World" in response.text
    assert "[DONE]" in response.text
def test_invalid_stream_key():
    response = client.post(
        "/chat/stream",
        headers=headers("wrong"),
        json={"message": "Hello"}
    )
    assert response.status_code == 403
def test_invalid_messages_key():
    session = create_session()
    response = client.get(
        f"/sessions/{session}/messages",
        headers=headers("wrong")
    )
    assert response.status_code == 403
