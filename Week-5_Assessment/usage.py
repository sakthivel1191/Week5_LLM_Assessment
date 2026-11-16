import json
import os
from datetime import date
from threading import Lock
USAGE_FILE = "usage.jsonl"
MODEL_PRICING = {
    "openai/gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60
    },
    "anthropic/claude-sonnet-4.5": {
        "input": 3.00,
        "output": 15.00
    }
}
DEFAULT_PRICING = {
    "input": float(os.getenv("INPUT_PRICE", "0.15")),
    "output": float(os.getenv("OUTPUT_PRICE", "0.60"))
}
_usage_lock = Lock()
def get_model_pricing(model):
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    return DEFAULT_PRICING
def save_usage(model, input_tokens, output_tokens):
    pricing = get_model_pricing(model)
    cost = (
        input_tokens * pricing["input"]
        + output_tokens * pricing["output"]
    ) / 1_000_000
    data = {
        "date": str(date.today()),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost": round(cost, 8)
    }
    with _usage_lock:
        with open(USAGE_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(data) + "\n")
    return data
def get_today_usage():
    total_tokens = 0
    total_cost = 0.0
    if not os.path.exists(USAGE_FILE):
        return {
            "total_tokens": 0,
            "total_cost": 0.0
        }
    with _usage_lock:
        with open(USAGE_FILE, encoding="utf-8") as file:
            for line in file:
                try:
                    item = json.loads(line)

                    if item.get("date") == str(date.today()):
                        total_tokens += item.get("total_tokens", 0)
                        total_cost += item.get("cost", 0)

                except (json.JSONDecodeError, TypeError):
                    continue
    return {
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 6)
    }
