
"""
Simple Research Orchestrator - Unidirectional Query to Report

Takes a user query and produces a final markdown research report.
No multi-turn conversations, just direct query -> research -> report.
"""

from typing_extensions import Literal
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

from src.master_state import MasterResearchState
from src.retrieval_agent import retrieval_agent
from src.prompts import final_report_generation_prompt
from src.utils import get_today_str

# Initialize model for synthesis
orchestrator_model = init_chat_model(model="openai:gpt-4o-mini", temperature=0)

def research_node(state: MasterResearchState) -> dict:
    """
    Direct research execution - no scoping, just research.
    Takes user query and runs full retrieval workflow.
    """
    
    # Get the user's query from messages
    user_query = state["messages"][-1].content
    
    # Execute retrieval workflow directly with the query
    retrieval_input = {
        "retriever_messages": [HumanMessage(content=user_query)],
        "tool_call_iterations": 0,
        "retrieval_topic": user_query,
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
        "tool_call_iterations": len(retrieval_result.get("retriever_messages", []))
    }



def synthesis_node(state: MasterResearchState) -> dict:
    """
    Create final markdown research report.
    """
    
    compressed_notes = state.get("compressed_notes", "")
    user_query = state["messages"][-1].content if state.get("messages") else "Research Query"
    
    # Generate final markdown report
    try:
        final_response = orchestrator_model.invoke([
            HumanMessage(content=final_report_generation_prompt.format(
                date=get_today_str(),
                research_topic=user_query,
                research_findings=compressed_notes
            ))
        ])
        
        response_content = final_response.content
    except Exception as e:
        # Fallback if synthesis fails
        response_content = f"""# Research Report: {user_query}

**Generated on:** {get_today_str()}

## Findings

{compressed_notes}

---
*This report was compiled from space biology research literature in our knowledge base.*"""

    return {
        "messages": [AIMessage(content=response_content)]
    }
