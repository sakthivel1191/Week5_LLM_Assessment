import json
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from tools import (
    get_order_status,
    get_customer,
    calculate_discount,
    execute_tool,
    run_tool_loop
)
def make_tool_call(
    name,
    arguments,
    tool_id="call_1"
):
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments)
        )
    )
def test_get_order_status():
    result = get_order_status("1001")
    assert result["status"] == "Shipped"
def test_unknown_order():
    result = get_order_status("9999")
    assert result == {
        "error": "Order 9999 not found"
    }
def test_get_customer():
    result = get_customer("1")
    assert result["name"] == "Rahul"
def test_unknown_customer():
    result = get_customer("9999")
    assert result == {
        "error": "Customer not found"
    }
def test_calculate_discount():
    result = calculate_discount(1000, 10)
    assert result["discount"] == 100
    assert result["final_amount"] == 900
def test_invalid_discount():
    result = calculate_discount(-100, 20)
    assert result== {"error": "Invalid argument"}
def test_invalid_percentage():
    result = calculate_discount(1000, 150)

    assert result == {"error": "Invalid argument"}
def test_unknown_tool():
    call = make_tool_call("unknown_tool",{})
    result = asyncio.run(execute_tool(call))
    data = json.loads(result["content"])
    assert result["tool_call_id"] == "call_1"
    assert "Unknown tool name" in data["error"]
def test_missing_argument():
    call = make_tool_call(
        "calculate_discount",
        {"amount": 1000}
    )
    result = asyncio.run(execute_tool(call))
    data = json.loads(result["content"])
    assert "error" in data
def test_malformed_json():
    call = SimpleNamespace(
        id="call_2",
        function=SimpleNamespace(
            name="get_order_status",
            arguments="{invalid json"
        )
    )
    result = asyncio.run(execute_tool(call))
    data = json.loads(result["content"])
    assert "error" in data
def test_parallel_tool_execution():
    calls = [
        make_tool_call(
            "get_order_status",
            {"order_id": "1001"},
            "call_1"
        ),
        make_tool_call(
            "get_customer",
            {"customer_id": "1"},
            "call_2"
        ),
        make_tool_call(
            "calculate_discount",
            {
                "amount": 2000,
                "percentage": 10
            },
            "call_3"
        )
    ]
    async def run():
        return await asyncio.gather(
            *[
                execute_tool(call)
                for call in calls
            ]
        )
    results = asyncio.run(run())
    assert len(results) == 3
    assert [
        result["tool_call_id"]
        for result in results
    ] == [
        "call_1",
        "call_2",
        "call_3"
    ]
def test_tool_loop():
    tool_calls = [
        make_tool_call(
            "get_order_status",
            {"order_id": "1001"},
            "call_1"
        ),
        make_tool_call(
            "get_customer",
            {"customer_id": "1"},
            "call_2"
        ),
        make_tool_call(
            "calculate_discount",
            {
                "amount": 2000,
                "percentage": 10
            },
            "call_3"
        )
    ]
    class MockAssistantMessage:
        def __init__(self, calls=None, content=None):
            self.tool_calls = calls
            self.content = content
        def model_dump(self, exclude_none=True):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": []
            }
    first_message = MockAssistantMessage(
        calls=tool_calls
    )
    second_message = MockAssistantMessage(
        calls=None,
        content="All tools executed successfully."
    )
    mock_llm = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    side_effect=[
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=first_message
                                )
                            ]
                        ),
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=second_message
                                )
                            ]
                        )
                    ]
                )
            )
        )
    )
    messages = [
        {
            "role": "user",
            "content": "Check my order and customer."
        }
    ]
    answer = asyncio.run(
        run_tool_loop(
            messages,
            llm=mock_llm
        )
    )
    assert answer == "All tools executed successfully."
    tool_results = [
        message
        for message in messages
        if message.get("role") == "tool"
    ]
    assert len(tool_results) == 3
