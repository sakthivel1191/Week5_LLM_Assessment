Week 5 - LLM Application Development
Project Overview
Build a secure and reliable company chat service using OpenAI, Anthropic and FastAPI. The application supports streaming, tool calling, conversation memory, token tracking and API authentication.

Technologies Used
Python
OpenAI SDK
Anthropic SDK
FastAPI
Pydantic
tiktoken
Tenacity
asyncio
pytest
python-dotenv
Tasks Completed
Task 1: SDK Calls, Streaming and Cost
Integrated OpenAI and Anthropic APIs.
Implemented streaming responses.
Added token counting using tiktoken.
Implemented API usage and cost tracking.
Task 2: Reliability and Async
Implemented retries using Tenacity.
Added exponential backoff.
Used AsyncOpenAI and asyncio.gather.
Controlled concurrency using Semaphore.
Task 3: Tools and Conversation Memory
Created order, customer and discount tools.
Implemented function calling and parallel tool execution.
Added conversation history and sliding window.
Implemented summarisation and token budgets.
Task 4: Secured FastAPI Backend
Created REST API endpoints.
Added Pydantic validation and dependency injection.
Implemented API key authentication.
Configured CORS and error handling.
Added pytest test cases.
Installation
bash
pip install -r requirements.txt
Environment Variables
Create a .env file:
env
OPENAI_API_KEY=my_openai_key
ANTHROPIC_API_KEY=my_anthropic_key
API_SECRET_KEY=my_secret_key

Run Application
bash
uvicorn main:app --reload
Swagger Documentation
Open in your browser:
http://127.0.0.1:8000/docs
Run Tests
bash
pytest --cov=. --cov-report=term-missing
Security
API keys are stored in environment variables. The .env file must not be committed to GitHub.

Author
Individual Week 5 Assessment - LLM Application Development
