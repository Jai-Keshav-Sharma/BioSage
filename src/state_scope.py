
import operator 
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# ==== STATE DEFINITIONS ====

class AgentState(MessagesState):
    """
    Main state for the full multi-agent RAG system.
    Extends MessagesState with additional fields.
    """

    # Retrieval brief generated from user conversation history
    retrieval_brief: Optional[str]
    # Messages exchanges 
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed documents retrieved from vector DB
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed notes after any cleaning or formatting
    notes: Annotated[list[str], operator.add] = []
    # Final formatted report
    final_report: Optional[str]

# ==== STRUCTURED OUTPUT SCHEMAS ====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question."
    )
    question: str = Field(
        description="A question to ask the user to clarify the retrieval scope."
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided necessary information."
    )

class RetrievalQuestion(BaseModel):
    """Schema for structured retrieval brief generation."""

    retrieval_brief: str = Field(
        description="A retrieval question that will be used to guide the retrieval."
    )
