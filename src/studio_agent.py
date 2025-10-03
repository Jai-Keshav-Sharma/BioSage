
"""
LangGraph Studio Entry Point for Space Biology Knowledge Engine

This file provides the graph object that LangGraph Studio expects.
"""

from langgraph.checkpoint.memory import InMemorySaver
from src.master_research_workflow import create_master_research_workflow

# Create the graph with checkpointer for Studio
def create_studio_graph():
    """Create graph compatible with LangGraph Studio."""

    # Create checkpointer for conversation memory
    checkpointer = InMemorySaver()

    # Create and compile the workflow with checkpointer
    workflow = create_master_research_workflow()

    # Note: If your workflow is already compiled, we need to rebuild with checkpointer
    from langgraph.graph import StateGraph, START, END
    from src.master_state import MasterResearchState
    from src.research_orchestrator import (
        research_orchestrator,
        scoping_node, 
        retrieval_node,
        synthesis_node,
        route_next_step
    )

    # Rebuild workflow with checkpointer
    studio_workflow = StateGraph(MasterResearchState)

    # Add nodes
    studio_workflow.add_node("orchestrator", research_orchestrator)
    studio_workflow.add_node("scoping", scoping_node)
    studio_workflow.add_node("retrieval", retrieval_node)
    studio_workflow.add_node("synthesis", synthesis_node)

    # Add edges
    studio_workflow.add_edge(START, "orchestrator")
    studio_workflow.add_conditional_edges(
        "orchestrator",
        route_next_step,
        {
            "scoping": "scoping",
            "retrieval": "retrieval", 
            "synthesis": "synthesis",
            "__end__": END
        }
    )
    studio_workflow.add_edge("scoping", "orchestrator")
    studio_workflow.add_edge("retrieval", "orchestrator") 
    studio_workflow.add_edge("synthesis", END)

    return studio_workflow.compile(checkpointer=checkpointer)

# Create the graph instance that Studio expects
graph = create_studio_graph()
