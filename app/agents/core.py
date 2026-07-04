from openai import AsyncOpenAI
from langsmith import traceable
from langsmith.wrappers import wrap_openai

from app.config import settings
from app.agents.memory import memory_store
from app.agents.schema import AgentResponse

client = wrap_openai(AsyncOpenAI(api_key=settings.openai_api_key))


SYSTEM_PROMPT = """
You are a helpful WhatsApp assistant.

Reply naturally.

Keep replies short.

Return ONLY the structured response.
"""


@traceable(name="run_agent")
async def run_agent(user_message: str, thread_id: str = "") -> AgentResponse:
    history = memory_store.get(thread_id) if thread_id else []

    input_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message},
    ]

    response = await client.responses.parse(
        model=settings.openai_model,
        input=input_messages,
        text_format=AgentResponse,
    )

    result = response.output_parsed

    if thread_id and result:
        memory_store.append(thread_id, "user", user_message)
        memory_store.append(thread_id, "assistant", result.reply)

    return result
