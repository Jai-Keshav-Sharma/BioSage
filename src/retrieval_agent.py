
"""Retrieval Agent Implementation.

This module implements a retrieval agent that can perform iterative
searches and synthesis to answer complex retrieval questions related to Space biology research.
"""

from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model

import src
from src.state_retrieval import RetrieverState, RetrieverOutputState
from src.prompts import retrieval_agent_prompt, compress_research_human_message, compress_research_system_prompt
from src.utils import search_research_papers, search_specific_topic, think_tool, search_nasa_web, get_today_str

# ==== CONFIGURATION ====

# Set up tools and model binding
tools = [search_research_papers, search_specific_topic, search_nasa_web, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# initialize models 
model = init_chat_model(model="groq:llama-3.3-70b-versatile")
model_with_tools = model.bind_tools(tools)
summarization_model = init_chat_model(model="groq:llama-3.3-70b-versatile")
compress_model = init_chat_model(model="groq:llama-3.3-70b-versatile")

# ==== AGENT NODES ====

def llm_call(state: RetrieverState):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    return {
        "retriever_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=retrieval_agent_prompt)] + state["retriever_messages"],
            )
        ]
    }

def tool_node(state: RetrieverState):
    """
    Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    tool_calls = state["retriever_messages"][-1].tool_calls

    # Execute all tool calls 
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"retriever_messages": tool_outputs}

def compress_retrieval(state: RetrieverState):
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """

    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("retriever_messages", []) + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages)

    # Extract raw notes from tool and AI messages
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["retriever_messages"],
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_notes": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ==== ROUTING LOGIC ====

def should_continue(state: RetrieverState) -> Literal["tool_node", "compress_retrieval"]:
    """Determine whether to continue retrieval process or provide final answer.

    Determines whether the agent should continue the retrieval loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress retrieval findings
    """

    messages = state["retriever_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool_call, continue to tool execution 
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we have the final answer
    return "compress_retrieval"

# ==== GRAPH CONSTRUCTION ====

# Build the agent workflow 
agent_builder = StateGraph(RetrieverState, output_schema=RetrieverOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_retrieval", compress_retrieval)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call", 
    should_continue, 
    {
        "tool_node": "tool_node", # Continue retrieval
        "compress_retrieval": "compress_retrieval" # Provide final answer
    }
)
agent_builder.add_edge("tool_node", "llm_call")
agent_builder.add_edge("compress_retrieval", END)

# Compile the graph
retrieval_agent = agent_builder.compile()
