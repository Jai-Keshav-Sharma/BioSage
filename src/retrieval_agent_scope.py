
"""
User Clarification and Research Brief Generation

This module implements the scoping phase of the retrieval workflow, where we:
1. Assess if the user's request needs clarification. 
2. Generate a detailed research brief from the conversation. 

The workflow uses structured outputs to make deterministic decisions about 
whether sufficient context exists to proced with research. 
"""

from datetime import datetime
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from src.prompts import clarify_with_user_instructions, transform_messages_into_retrieval_topic_prompt
from src.state_scope import AgentState, ClarifyWithUser, RetrievalQuestion

from dotenv import load_dotenv
load_dotenv()

# ==== UTILITY FUNCTIONS ====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %#d, %Y")


# ==== CONFIGURATOIN ====

# Initialize model
model = init_chat_model(model="groq:meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

# ==== WORKFLOW NODES ====

def clarify_with_user(state: AgentState) -> Command[Literal["write_retrieval_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with retrieval.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """

    # Set up structured output model
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]),
            date=get_today_str()
        ))
    ])

    # Route based on clssarification need
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_retrieval_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )

def write_retrieval_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive retrieval brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """

    # Set up structured output model
    structured_output_model = model.with_structured_output(RetrievalQuestion)

    # Generate retrieval brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_retrieval_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return {
        "retrieval_brief": response.retrieval_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.retrieval_brief}.")]
    }

# ==== GRAPH CONSTRUCTION ====

# Build the scoping nodes
retrieval_builder = StateGraph(AgentState)

# Add workflow nodes
retrieval_builder.add_node(clarify_with_user, name="clarify_with_user")
retrieval_builder.add_node(write_retrieval_brief, name="write_retrieval_brief")

# Add workflow edges
retrieval_builder.add_edge(START, "clarify_with_user")
retrieval_builder.add_edge("write_retrieval_brief", END)

# Compile the workflow
scope_research = retrieval_builder.compile()
