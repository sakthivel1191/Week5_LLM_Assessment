import os
import time
import asyncio
import logging
import tiktoken
from dotenv import load_dotenv
from openai import (OpenAI,AsyncOpenAI)
from anthropic import Anthropic
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception,
    before_sleep_log
)
from usage import save_usage

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    API_KEY = "test-placeholder"
MODEL = os.getenv("MODEL","openai/gpt-4o-mini")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL","anthropic/claude-sonnet-4.5")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
client = OpenAI(
    api_key=API_KEY,
    base_url=OPENROUTER_BASE_URL
)
async_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=OPENROUTER_BASE_URL
)
anthropic_client = Anthropic(
    api_key=API_KEY,
    base_url="https://openrouter.ai/api"
)
def count_tokens(messages):
    """
    Count tokens in conversation messages
    using tiktoken.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    total = 2
    for message in messages:
        total += 4
        for key in ("role", "content"):
            value = message.get(key, "")
            if isinstance(value, str):
                total += len(
                    encoding.encode(value)
                )
            elif value is not None:
                total += len(
                    encoding.encode(str(value))
                )
    return total
def check_token_limit(messages, limit=3000):
    """
    Refuse requests above the configured token limit.
    """
    tokens = count_tokens(messages)
    if tokens > limit:
        raise ValueError(
            f"Token limit exceeded: {tokens}/{limit}"
        )
    return tokens
def ask_openai(prompt):
    """
    Send a non-streaming request using OpenAI SDK.
    """
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    check_token_limit(messages)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000
    )
    answer = (
        response.choices[0].message.content or ""
    )
    elapsed = time.perf_counter() - start
    if response.usage:
        usage = save_usage(
            MODEL,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
        print("Token usage:", usage)
    print("Answer:", answer)
    print("Model:", MODEL)
    print("Response time:",round(elapsed, 2),"seconds")
    return answer
def ask_claude(prompt, system):
    """
    Send a Claude request using the official
    Anthropic Python SDK and OpenRouter endpoint.
    """
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    check_token_limit(
        [
            {
                "role": "system",
                "content": system
            },
            *messages
        ]
    )
    start = time.perf_counter()
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        system=system,
        messages=messages,
        max_tokens=1000
    )
    answer = "".join(
        block.text
        for block in response.content
        if block.type == "text"
    )
    elapsed = time.perf_counter() - start
    usage = save_usage(
        CLAUDE_MODEL,
        response.usage.input_tokens,
        response.usage.output_tokens
    )
    print("Answer:", answer)
    print("Model:", CLAUDE_MODEL)
    print("Token usage:", usage)
    print("Response time:",round(elapsed, 2),"seconds")
    return answer
def stream_answer(prompt):
    """
    Stream OpenAI-compatible model output.
    Measures time to first token.
    """
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    check_token_limit(messages)
    start = time.perf_counter()
    first_token = True
    full_answer = ""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=1000,
        stream=True,
        stream_options={
            "include_usage": True
        }
    )
    try:
        for chunk in stream:
            if chunk.choices:
                content = (chunk.choices[0].delta.content)
                if content:
                    if first_token:
                        elapsed = (time.perf_counter() - start)
                        print("\nTime to first token:",round(elapsed, 2),"seconds")
                        first_token = False
                    full_answer += content
                    print(content,end="",flush=True)
            if chunk.usage:
                save_usage(
                    MODEL,
                    chunk.usage.prompt_tokens,
                    chunk.usage.completion_tokens
                )
    finally:
        stream.close()
    print()
    return full_answer
def measure_non_streaming(prompt):
    """
    Measure complete non-streaming response time.
    """
    start = time.perf_counter()
    answer = ask_openai(prompt)
    elapsed = time.perf_counter() - start
    print("Non-streaming total time:",round(elapsed, 2),"seconds")
    return answer
def retryable_error(error):
    """
    Retry only HTTP 429 and HTTP 5xx errors.
    """
    status = getattr(error,"status_code",None)
    return (
        status == 429
        or (
            status is not None
            and 500 <= status <= 599
        )
    )
@retry(
    wait=wait_exponential(
        multiplier=1,
        min=1,
        max=10
    ),
    stop=stop_after_attempt(5),
    retry=retry_if_exception(retryable_error),
    before_sleep=before_sleep_log(
        logger,
        logging.WARNING
    ),
    reraise=True
)
def reliable_call(prompt):
    """
    Retry failed requests for 429 and 5xx.
    """
    return ask_openai(prompt)
async def async_question(prompt):
    """
    Send asynchronous OpenAI-compatible request.
    """
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    check_token_limit(messages)
    response = await async_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500
    )
    answer = (
        response.choices[0].message.content or ""
    )
    if response.usage:
        save_usage(
            MODEL,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
    return answer
async def sequential_questions(questions):
    """
    Execute questions one by one.
    """
    results = []
    for question in questions:
        try:
            answer = await async_question(question)
            results.append(answer)
        except Exception as error:
            results.append({
                "error": str(error)
            })
    return results
async def parallel_questions(questions):
    """
    Execute questions concurrently.

    Maximum concurrency = 3.
    """
    semaphore = asyncio.Semaphore(3)
    async def run(question):
        async with semaphore:
            try:
                return await async_question(
                    question
                )
            except Exception as error:
                return {
                    "error": str(error)
                }
    results = await asyncio.gather(
        *[
            run(question)
            for question in questions
        ]
    )
    return results