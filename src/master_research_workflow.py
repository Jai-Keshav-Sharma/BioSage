
"""
Simple Research Workflow

Unidirectional: Query -> Research -> Report
"""

from langgraph.graph import StateGraph, START, END
from src.master_state import MasterResearchState
from src.research_orchestrator import (
    research_node,
    synthesis_node
)

def create_master_research_workflow():
    """
    Create simple unidirectional research workflow.
    """
    
    # Initialize workflow graph
    workflow = StateGraph(MasterResearchState)
    
    # Add workflow nodes
    workflow.add_node("research", research_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # Linear flow: research -> synthesis
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()# Create the master workflow
master_research_agent = create_master_research_workflow()
