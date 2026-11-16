import json
import asyncio
from llm import async_client, MODEL
orders = {
    "1001": {
        "status": "Shipped",
        "amount": 2000
    },
    "1002": {
        "status": "Processing",
        "amount": 3000
    }
}
customers = {
    "1": {
        "name": "Rahul",
        "email": "rahul@gmail.com"
    },
    "2": {
        "name": "Priya",
        "email": "priya@gmail.com"
    }
}
def get_order_status(order_id):
    order = orders.get(str(order_id))
    if not order:
        return {
            "error": f"Order {order_id} not found"
        }
    return order
def get_customer(customer_id):
    customer = customers.get(str(customer_id))
    if not customer:
        return {
            "error": "Customer not found"
        }
    return customer
def calculate_discount(amount, percentage):
    if (
        isinstance(amount, bool)
        or isinstance(percentage, bool)
        or not isinstance(amount, (int, float))
        or not isinstance(percentage, (int, float))
        or amount < 0
        or percentage < 0
        or percentage > 100
    ):
        return {
            "error": "Invalid argument"
        }
    discount = amount * percentage / 100
    return {
        "original_amount": amount,
        "discount": discount,
        "final_amount": amount - discount
    }
tool_functions = {
    "get_order_status": get_order_status,
    "get_customer": get_customer,
    "calculate_discount": calculate_discount
}
tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Get order status and amount using order ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID"
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": (
                "Get customer details using customer ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer ID"
                    }
                },
                "required": ["customer_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_discount",
            "description": (
                "Calculate discount and final amount."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number"
                    },
                    "percentage": {
                        "type": "number"
                    }
                },
                "required": ["amount", "percentage"],
                "additionalProperties": False
            }
        }
    }
]
async def execute_tool(tool_call):
    name = tool_call.function.name
    tool_id = tool_call.id
    try:
        if name not in tool_functions:
            result = {
                "error": f"Unknown tool name: {name}"
            }
        else:
            args = json.loads(tool_call.function.arguments)
            if not isinstance(args, dict):
                raise ValueError(
                    "Tool arguments must be a JSON object"
                )
            result = await asyncio.to_thread(
                tool_functions[name],
                **args
            )
    except Exception as error:
        result = {
            "error": str(error)
        }
    return {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": json.dumps(result)
    }
async def run_tool_loop(
    messages,
    max_rounds=5,
    llm=None
):
    if llm is None:
        llm = async_client
    for round_number in range(max_rounds):
        response = await llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto"
        )
        message = response.choices[0].message
        if not message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content or ""
            })
            return message.content or ""
        messages.append(
            message.model_dump(exclude_none=True))
        results = await asyncio.gather(
            *[
                execute_tool(call)
                for call in message.tool_calls
            ]
        )
        messages.extend(results)
        print(
            f"Round {round_number + 1}: "
            f"{len(results)} tool calls executed"
        )
    return "Maximum tool rounds reached"
