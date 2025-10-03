
"""
State Definitions and Pydantic Schemas for Retrieval Agent

This module defines the state objects and structured schemas used for
the retrieval agent workflow, including retrieval state management and output schemas.
"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ==== STATE DEFINITIONS ====

class RetrieverState(TypedDict):
    """
    State for the retrieval agent containing message history and retrieval metadata.

    This state tracks the retriever's conversation, iteration count for limiting
    tool calls, the retrieval topic being investigated
    and raw retrieval notes for detailed analysis.
    """

    retriever_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    retrieval_topic: str
    compressed_notes: str
    raw_notes: Annotated[List[str], operator.add]

class RetrieverOutputState(TypedDict):
    """
    Output state for the retriever agent containing final retrieval results.

    This represents the final output of the retrieval process with 
    all raw notes from the retrieval process.
    """
    raw_notes: Annotated[List[str], operator.add]
    compressed_notes: str
    retriever_messages: Annotated[Sequence[BaseMessage], add_messages]

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decisions during scoping phase."""
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the retrieval scope",
    )
    verification: str = Field(
        description="Verify message that we will start retrieval after the user has provided the necessary information.",
    )

class RetrievalQuestion(BaseModel):
    """Schema for retrieval question generation."""
    retrieval_question: str = Field(
        description="A retrieval question that will be used to guide the retrieval.",
    )

class Summary(BaseModel):
    """Schema for retrieved content summarization."""
    summary: str = Field(description="Concise summary of the retrieved content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the retrieved content")
