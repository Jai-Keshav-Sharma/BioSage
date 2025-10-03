
"""
Master Research Workflow

Complete orchestration following the full agent pattern from reference.
"""

from langgraph.graph import StateGraph, START, END
from src.master_state import MasterResearchState
from src.research_orchestrator import (
    research_orchestrator,
    scoping_node, 
    retrieval_node,
    synthesis_node,
    route_next_step
)

def create_master_research_workflow():
    """
    Create the complete research workflow following reference architecture.
    """

    # Initialize workflow graph
    workflow = StateGraph(MasterResearchState)

    # Add orchestrator and workflow nodes
    workflow.add_node("orchestrator", research_orchestrator)
    workflow.add_node("scoping", scoping_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("synthesis", synthesis_node)

    # Start with orchestrator
    workflow.add_edge(START, "orchestrator")

    # Route from orchestrator to appropriate workflow step
    workflow.add_conditional_edges(
        "orchestrator",
        route_next_step,
        {
            "scoping": "scoping",
            "retrieval": "retrieval", 
            "synthesis": "synthesis",
            "__end__": END
        }
    )

    # After each step, return to orchestrator for next routing decision
    workflow.add_edge("scoping", "orchestrator")
    workflow.add_edge("retrieval", "orchestrator") 
    workflow.add_edge("synthesis", END)

    return workflow.compile()

# Create the master workflow
master_research_agent = create_master_research_workflow()
