
"""
Research Orchestrator - Main Workflow Controller

Based on the full agent orchestration pattern from the reference.
Coordinates between scoping and retrieval phases.
"""

from typing_extensions import Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END

from src.master_state import MasterResearchState
from src.retrieval_agent_scope import scope_research  
from src.retrieval_agent import retrieval_agent
from src.prompts import final_report_generation_prompt
from src.utils import get_today_str

# Initialize orchestrator model
orchestrator_model = init_chat_model(model="openai:gpt-4o-mini", temperature=0)

def research_orchestrator(state: MasterResearchState) -> dict:
    """
    Main orchestrator that decides the next step in research process.

    Following the reference pattern for workflow coordination.
    """

    # Check current state to determine next action
    has_brief = bool(state.get("retrieval_brief"))
    has_research = bool(state.get("compressed_notes"))

    if not has_brief:
        return {"next_step": "scoping"}
    elif has_brief and not has_research:
        return {"next_step": "retrieval"}
    else:
        return {"next_step": "synthesis"}

def scoping_node(state: MasterResearchState) -> dict:
    """
    Execute scoping workflow and update master state.

    Calls your existing scoping agent and integrates results.
    """

    # Prepare input for scoping workflow
    scoping_input = {
        "messages": state.get("messages", [])
    }

    # Execute scoping workflow
    scoping_result = scope_research.invoke(scoping_input)

    # Update master state with scoping results
    updates = {
        "messages": scoping_result.get("messages", [])
    }

    # Check if we got a retrieval brief
    if scoping_result.get("retrieval_brief"):
        updates["retrieval_brief"] = scoping_result["retrieval_brief"]
        updates["next_step"] = "retrieval"
    else:
        updates["next_step"] = "complete"  # Clarification question was asked

    return updates

def retrieval_node(state: MasterResearchState) -> dict:
    """
    Execute retrieval workflow and update master state.

    Calls your existing retrieval agent with the research brief.
    """

    # Prepare input for retrieval workflow
    retrieval_input = {
        "retriever_messages": [HumanMessage(content=state["retrieval_brief"])],
        "tool_call_iterations": 0,
        "retrieval_topic": state["retrieval_brief"],
        "compressed_notes": "",
        "raw_notes": []
    }

    # Execute retrieval workflow  
    retrieval_result = retrieval_agent.invoke(retrieval_input)

    # Update master state with retrieval results
    return {
        "retriever_messages": retrieval_result.get("retriever_messages", []),
        "compressed_notes": retrieval_result.get("compressed_notes", ""),
        "raw_notes": retrieval_result.get("raw_notes", []),
        "tool_call_iterations": len(retrieval_result.get("retriever_messages", [])),
        "next_step": "synthesis"
    }

def synthesis_node(state: MasterResearchState) -> dict:
    """
    Create final synthesized response using the final_report_generation_prompt.

    Uses your existing prompt template for consistent formatting.
    """

    compressed_notes = state.get("compressed_notes", "")
    retrieval_brief = state.get("retrieval_brief", "Unknown research topic")

    # Use your existing final report generation prompt
    try:
        final_response = orchestrator_model.invoke([
            HumanMessage(content=final_report_generation_prompt.format(
                date=get_today_str(),
                research_topic=retrieval_brief,
                research_findings=compressed_notes
            ))
        ])

        response_content = final_response.content
    except Exception as e:
        # Fallback if synthesis fails
        response_content = f"""Based on space biology research literature:

        **Research Topic:** {retrieval_brief}

        **Findings:**
        {compressed_notes}

        This information was compiled from multiple research papers in our space biology knowledge base."""

    return {
        "messages": [AIMessage(content=response_content)],
        "next_step": "complete"
    }

# Routing function following reference pattern
def route_next_step(state: MasterResearchState) -> Literal["scoping", "retrieval", "synthesis", "__end__"]:
    """Route to next step based on workflow state."""

    next_step = state.get("next_step", "scoping")

    if next_step == "complete":
        return "__end__"
    else:
        return next_step
