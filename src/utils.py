
"""
Retrieval Utilities and Tools. 

This module provides utility functions and tools to support the retrieval process.
"""

from pathlib import Path
from datetime import datetime
import platform
from typing_extensions import Annotated, List

from langchain.chat_models import init_chat_model 
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from tavily import TavilyClient  

from src.config import QDRANT_PATH, COLLECTION_NAME, EMBEDDING_MODEL
from src.state_retrieval import Summary
from src.prompts import summarize_retrieved_content_prompt, summarize_webpage_prompt

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    # Windows-compatible date format
    if platform.system() == "Windows":
        date_str = datetime.now().strftime("%a %b %d, %Y")
        parts = date_str.split()
        if len(parts) >= 3 and parts[2].startswith('0'):
            parts[2] = parts[2][1:]
        return ' '.join(parts)
    else:
        return datetime.now().strftime("%a %b %-d, %Y")

def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()

def get_vector_store():
    """Initialize and return the Qdrant vector store."""
    
    from src.create_vectorstore import CLIPEmbeddings
    embeddings = CLIPEmbeddings(model_name=EMBEDDING_MODEL)

    client = QdrantClient(url="http://localhost:6333")

    vector_store = Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    return vector_store

def format_source_metadata(doc, index: int) -> str:
    """Format source information using rich metadata."""
    metadata = doc.metadata

    # Extract clean information from your rich metadata
    pmc_id = metadata.get('pmc_id', 'Unknown PMC')
    title = metadata.get('title', metadata.get('filename', 'Unknown title'))
    organisms = metadata.get('organisms', [])
    research_types = metadata.get('research_types', [])

    # Clean up title (remove file extension artifacts)
    if title.endswith('.pdf'):
        title = title[:-4]

    # Format organisms and research types
    org_str = f" | Organisms: {', '.join(organisms[:2])}" if organisms and organisms != ['unknown'] else ""
    type_str = f" | Type: {', '.join(research_types[:2])}" if research_types and research_types != ['general'] else ""

    return f"[{index}] {pmc_id}: {title[:60]}...{org_str}{type_str}"

# Initialize summarization model
summarization_model = init_chat_model(model="openai:gpt-4o-mini")

def summarize_retrieved_content(retrieved_content: str) -> str:
    """Summarize retrieved research content using structured output.

    Args:
        retrieved_content: Raw retrieved content from vector search

    Returns:
        Formatted summary with key excerpts
    """
    try:
        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(Summary)

        # Generate summary
        summary = structured_model.invoke([
            HumanMessage(content=summarize_retrieved_content_prompt.format(
                date=get_today_str(),
                retrieved_content=retrieved_content
            ))
        ])

        # Format summary with clear structure
        formatted_summary = (
            f"**Summary:**\n{summary.summary}\n\n"
            f"**Key Excerpts:**\n{summary.key_excerpts}"
        )

        return formatted_summary

    except Exception as e:
        print(f"Failed to summarize content: {str(e)}")
        # Fallback: return first 2000 chars if summarization fails
        return retrieved_content[:2000] + "..." if len(retrieved_content) > 2000 else retrieved_content

# ===== RETRIEVAL TOOLS =====

@tool
def search_research_papers(query: str, max_results: int = 5) -> str:
    """
    Search through space biology research papers for relevant information.

    Args:
        query: The search query describing what information you're looking for
        max_results: Maximum number of results to return (default: 5, max: 15)

    Returns:
        String containing research findings (automatically summarized if content is too long)
    """

    max_results = min(max_results, 15)

    try:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(query, k=max_results)

        if not docs:
            return f"No relevant research papers found for query: '{query}'"

        # Combine all content
        all_content = []
        sources = []

        for i, doc in enumerate(docs, 1):
            # Use rich metadata for source formatting
            source_info = format_source_metadata(doc, i)
            sources.append(source_info)
            all_content.append(f"Research Finding {i}:\n{doc.page_content}")

        combined_content = "\n\n".join(all_content)
        source_list = "\n".join(sources)

        # Auto-summarize if content is too long (> 3000 characters)
        if len(combined_content) > 3000:
            summary = summarize_retrieved_content(combined_content)
            return f"{summary}\n\n**Sources:**\n{source_list}"
        else:
            return f"{combined_content}\n\n**Sources:**\n{source_list}"

    except Exception as e:
        return f"Error searching research papers: {str(e)}"


@tool
def search_specific_topic(topic: str, keywords: List[str], max_results: int = 8) -> str:
    """
    Search for specific topics using targeted keywords for more focused results.

    Args:
        topic: The main topic you're researching
        keywords: List of specific keywords to focus the search
        max_results: Maximum number of results (default: 8, max: 12)

    Returns:
        String containing focused research findings (automatically summarized if content is too long)
    """

    max_results = min(max_results, 12)

    # Simple query construction
    keyword_str = " ".join(keywords)
    focused_query = f"{topic} {keyword_str}"

    try:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(focused_query, k=max_results)

        if not docs:
            return f"No specific research found for topic '{topic}' with keywords: {keywords}"

        # Combine all content
        all_content = []
        sources = []

        for i, doc in enumerate(docs, 1):
            # Use rich metadata for source formatting
            source_info = format_source_metadata(doc, i)
            sources.append(source_info)
            all_content.append(f"{topic} Finding {i}:\n{doc.page_content}")

        combined_content = "\n\n".join(all_content)
        source_list = "\n".join(sources)

        # Auto-summarize if content is too long (> 3000 characters)
        if len(combined_content) > 3000:
            summary = summarize_retrieved_content(combined_content)
            return f"{summary}\n\n**Sources:**\n{source_list}"
        else:
            return f"{combined_content}\n\n**Sources:**\n{source_list}"

    except Exception as e:
        return f"Error in focused search: {str(e)}"


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the retrieval workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


@tool
def search_nasa_web(query: str, max_results: int = 3) -> str:
    """
    Search official NASA websites (nasa.gov domains) for authoritative space-related information.
    
    Args:
        query: The search query for NASA websites
        max_results: Maximum number of results to return (default: 3, max: 5)
        
    Returns:
        String containing summarized findings from NASA websites
    """
    import os
    
    max_results = min(max_results, 5)
    
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY not found in environment"
            
        client = TavilyClient(api_key=api_key)
        
        # Search with NASA domain restriction
        results = client.search(
            query=query,
            max_results=max_results,
            include_domains=["nasa.gov"],
            search_depth="advanced", 
            include_raw_content=True
        )
        
        if not results.get('results'):
            return f"No results found on official NASA websites for: '{query}'"
        
        # Process and summarize results
        all_content = []
        sources = []
        
        for i, result in enumerate(results['results'], 1):
            url = result.get('url', 'Unknown URL')
            title = result.get('title', 'Untitled')
            content = result.get('content', '')
            
            if content:
                # Use structured summarization (same pattern as existing tools)
                summary = summarize_nasa_webpage(content)
                all_content.append(f"NASA Finding {i}:\n{summary}")
                sources.append(f"[{i}] {title}: {url}")
        
        if not all_content:
            return f"No content found in NASA search results for: '{query}'"
        
        combined_content = "\n\n".join(all_content)
        source_list = "\n".join(sources)
        
        return f"{combined_content}\n\n**NASA Sources:**\n{source_list}"
        
    except Exception as e:
        return f"Error searching NASA websites: {str(e)}"


def summarize_nasa_webpage(content: str) -> str:
    """Summarize NASA webpage content using structured output (same pattern as existing summarization)."""
    try:
        # Use existing structured output pattern
        structured_model = summarization_model.with_structured_output(Summary)
        
        # Generate structured summary using NASA-specific prompt
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=content,
                date=get_today_str()
            ))
        ])
        
        # Format using the same pattern as existing tools
        formatted_summary = (
            f"**Summary:**\n{summary.summary}\n\n"
            f"**Key Excerpts:**\n{summary.key_excerpts}"
        )
        
        return formatted_summary
        
    except Exception as e:
        print(f"Failed to summarize NASA webpage: {str(e)}")
        # Fallback to truncated content
        return content[:800] + "..." if len(content) > 800 else content