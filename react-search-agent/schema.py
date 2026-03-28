from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response for with answer and sources"""

    answer: str = (Field(description="The agent answer to query"),)
    sources: list[Source] = Field(
        default_factory=list, description="List of sources to generate answers"
    )
