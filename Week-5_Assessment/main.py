import os
import uuid
import json
import secrets
from typing import List
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Header
)
from fastapi.responses import StreamingResponse,JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from llm import async_client, MODEL, count_tokens
from memory import Conversation
from usage import get_today_usage, save_usage
load_dotenv()

app = FastAPI(
    title="Company Chat Service",
    version="1.0.0"
)
sessions = {}
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "x-api-key"]
)
class Settings:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
def get_settings():
    return Settings()
def get_llm_client():
    return async_client
def get_store():
    return sessions
def verify_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings)
):
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key missing"
        )
    if not settings.api_key:
        raise HTTPException(
            status_code=500,
            detail="API authentication is not configured"
        )
    if not secrets.compare_digest(
        x_api_key,
        settings.api_key
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return True
class SessionResponse(BaseModel):
    session_id: str
class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=2000
    )
class ChatResponse(BaseModel):
    answer: str
class MessageResponse(BaseModel):
    messages: List[dict]
@app.exception_handler(HTTPException)
async def http_error_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail
        },
        headers=exc.headers
    )
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    errors = exc.errors()
    too_long = any(
        error.get("type") == "string_too_long"
        for error in errors
    )
    return JSONResponse(
        status_code=413 if too_long else 422,
        content={
            "error": (
                "Message too long"
                if too_long
                else "Invalid request"
            ),
            "details": errors
        }
    )
@app.get("/health")
async def health():
    return {
        "status": "ok"
    }
@app.post(
    "/sessions",
    response_model=SessionResponse
)
async def create_session(
    store: dict = Depends(get_store),
    auth: bool = Depends(verify_api_key)
):
    session_id = str(uuid.uuid4())
    store[session_id] = Conversation()
    return {
        "session_id": session_id
    }
@app.post(
    "/sessions/{session_id}/chat",
    response_model=ChatResponse
)
async def chat(
    session_id: str,
    request: ChatRequest,
    store: dict = Depends(get_store),
    llm=Depends(get_llm_client),
    auth: bool = Depends(verify_api_key)
):
    if session_id not in store:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    conversation = store[session_id]
    original_messages = conversation.messages.copy()
    try:
        answer = await conversation.chat(request.message,llm)
        return {
            "answer": answer
        }
    except Exception as error:
        conversation.messages = original_messages
        print("LLM error:", error)
        raise HTTPException(
            status_code=502,
            detail="LLM service failed"
        )
@app.get(
    "/sessions/{session_id}/messages",
    response_model=MessageResponse
)
async def get_messages(
    session_id: str,
    store: dict = Depends(get_store),
    auth: bool = Depends(verify_api_key)
):
    if session_id not in store:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    return {
        "messages": store[session_id].get_history()
    }

@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    llm=Depends(get_llm_client),
    auth: bool = Depends(verify_api_key)
):
    messages = [
        {
            "role": "user",
            "content": request.message
        }
    ]
    if count_tokens(messages) > 3000:
        raise HTTPException(
            status_code=413,
            detail="Token limit exceeded"
        )
    async def generate():
        input_tokens = None
        output_tokens = None
        try:
            stream = await llm.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=1000,
                stream=True,
                stream_options={"include_usage": True}
            )
            async for chunk in stream:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield ("data: "+ json.dumps({"content": content})+ "\n\n")
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens
            if input_tokens is not None:
                save_usage(
                    MODEL,
                    input_tokens,
                    output_tokens
                )
            yield "data: [DONE]\n\n"
        except Exception:
            yield ("data: "+ json.dumps({"error": "LLM service failed"})+ "\n\n")
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
@app.get("/usage")
async def usage(
    auth: bool = Depends(verify_api_key)
):
    return get_today_usage()
