from pydantic import BaseModel


class AgentResponse(BaseModel):
    reply: str
