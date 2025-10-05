
"""
Master State for Integrated Space Biology Research Workflow

Based on the full agent pattern but simplified for sequential execution.
"""

import operator
from typing_extensions import TypedDict, Annotated, List, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MasterResearchState(TypedDict):
    """
    Master state combining scoping and retrieval workflows.
    
    Based on the notebook reference pattern.
    """
    
    # === CORE CONVERSATION ===
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # === SCOPING RESULTS ===
    retrieval_brief: Optional[str]
    
    # === RETRIEVAL RESULTS ===
    retriever_messages: Annotated[Sequence[BaseMessage], add_messages] 
    tool_call_iterations: int
    compressed_notes: str
    raw_notes: Annotated[List[str], operator.add]
