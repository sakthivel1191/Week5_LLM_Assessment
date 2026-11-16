from llm import MODEL, count_tokens
from usage import save_usage

SYSTEM_BUDGET = 500
HISTORY_BUDGET = 3000
OUTPUT_BUDGET = 1000
RECENT_MESSAGES = 6
SYSTEM_PROMPT = (
    "You are a helpful company customer support assistant. "
    "Remember important information provided by the user."
)
class Conversation:
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content
        })
    def get_history(self):
        return self.messages.copy()
    def get_token_count(self):
        return count_tokens(self.messages)
    async def summarise(self, llm):
        if len(self.messages) <= RECENT_MESSAGES + 1:
            return
        system_message = self.messages[0]
        old_messages = self.messages[1:-RECENT_MESSAGES]
        recent_messages = self.messages[-RECENT_MESSAGES:]
        if not old_messages:
            return
        summary_request = [
            {
                "role": "system",
                "content": (
                    "You are a conversation summarisation assistant. "
                    "Summarise the older conversation while preserving "
                    "important facts, names, employee IDs, order IDs, "
                    "dates, preferences and user instructions. "
                    "Do not invent information. Keep the summary concise."
                )
            },
            {
                "role": "user",
                "content": str(old_messages)
            }
        ]
        response = await llm.chat.completions.create(
            model=MODEL,
            messages=summary_request,
            max_tokens=400
        )
        summary = (
            response.choices[0].message.content or ""
        )
        if response.usage:
            save_usage(
                MODEL,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
        self.messages = [
            system_message,
            {
                "role": "system",
                "content": (
                    "Important conversation summary: " + summary
                )
            }
        ] + recent_messages
        print("Older conversation summarised.")
    async def manage_budget(self, llm=None):
        if (
            llm is not None
            and (
                self.get_token_count() > HISTORY_BUDGET
                or len(self.messages) > RECENT_MESSAGES + 1
            )
        ):
            await self.summarise(llm)
        system_messages = [
            message
            for message in self.messages
            if message["role"] == "system"
        ]

        if count_tokens(system_messages) > SYSTEM_BUDGET:
            raise ValueError(
                "System prompt exceeds token budget"
            )
        while (
            self.get_token_count() > HISTORY_BUDGET
            and len(self.messages) > 2
        ):
            del self.messages[1]
        if self.get_token_count() > HISTORY_BUDGET:
            raise ValueError(
                "Conversation history exceeds token budget"
            )
    async def chat(self, message, llm):
        original_messages = self.messages.copy()
        try:
            self.add_message("user", message)
            await self.manage_budget(llm)
            response = await llm.chat.completions.create(
                model=MODEL,
                messages=self.get_history(),
                max_tokens=OUTPUT_BUDGET
            )
            answer = (response.choices[0].message.content or "")
            if response.usage:
                save_usage(
                    MODEL,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            self.add_message("assistant", answer)
            await self.manage_budget(llm)
            return answer
        except Exception:
            self.messages = original_messages
            raise
