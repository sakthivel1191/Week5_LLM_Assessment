from unittest.mock import Mock
import pytest
import llm
from tenacity import wait_none
def run_without_waiting(prompt):
    retry_call = llm.reliable_call.retry_with(
        wait=wait_none()
    )
    return retry_call(prompt)
class APIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
def test_retry_twice_then_success(monkeypatch):
    mock_call = Mock(
        side_effect=[
            APIError(429),
            APIError(429),
            "Success"
        ]
    )
    monkeypatch.setattr(
        llm,
        "ask_openai",
        mock_call
    )
    result = run_without_waiting("Hello")
    assert result == "Success"
    assert mock_call.call_count == 3
def test_401_not_retried(monkeypatch):
    mock_call = Mock(
        side_effect=APIError(401)
    )
    monkeypatch.setattr(
        llm,
        "ask_openai",
        mock_call
    )
    with pytest.raises(APIError):
        run_without_waiting("Hello")
    assert mock_call.call_count == 1
def test_server_error_retry(monkeypatch):
    mock_call = Mock(
        side_effect=[
            APIError(500),
            "Success"
        ]
    )
    monkeypatch.setattr(
        llm,
        "ask_openai",
        mock_call
    )
    result = run_without_waiting("Hello")
    assert result == "Success"
    assert mock_call.call_count == 2
def test_bad_request_not_retried(monkeypatch):
    mock_call = Mock(
        side_effect=APIError(400)
    )
    monkeypatch.setattr(
        llm,
        "ask_openai",
        mock_call
    )
    with pytest.raises(APIError):
        run_without_waiting("Hello")
    assert mock_call.call_count == 1
