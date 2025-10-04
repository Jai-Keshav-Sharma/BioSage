"""
Test script for the Space Biology Knowledge Engine with NASA Web Search integration.

This script tests the complete workflow including:
1. Database search capabilities  
2. NASA web search integration
3. Multi-agent research orchestration
4. Final report generation

Run this to verify everything is working correctly.
"""

import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append('src')

from src.master_research_workflow import master_research_agent
from src.master_state import MasterResearchState
from langchain_core.messages import HumanMessage

# Initialize rich console for beautiful output
console = Console()

def test_nasa_search_integration():
    """Test the NASA web search functionality directly."""
    console.print("\n🔍 [bold blue]Testing NASA Web Search Integration[/bold blue]")
    
    try:
        from src.utils import search_nasa_web
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description="Searching NASA websites...", total=None)
            
            # Test NASA search with proper tool invocation
            result = search_nasa_web.invoke({"query": "Mars biology research", "max_results": 2})
            
        console.print("\n✅ [green]NASA Search Test Result:[/green]")
        console.print(Panel(result[:500] + "..." if len(result) > 500 else result, 
                          title="NASA Search Output", 
                          border_style="green"))
        return True
        
    except Exception as e:
        console.print(f"\n❌ [red]NASA Search Test Failed:[/red] {str(e)}")
        return False

def test_database_search():
    """Test the vector database search functionality."""
    console.print("\n📚 [bold blue]Testing Database Search[/bold blue]")
    
    try:
        from src.utils import search_research_papers
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description="Searching research database...", total=None)
            
            # Test database search with proper tool invocation
            result = search_research_papers.invoke({"query": "microgravity effects on plants", "max_results": 2})
            
        console.print("\n✅ [green]Database Search Test Result:[/green]")
        console.print(Panel(result[:500] + "..." if len(result) > 500 else result,
                          title="Database Search Output",
                          border_style="green"))
        return True
        
    except Exception as e:
        console.print(f"\n❌ [red]Database Search Test Failed:[/red] {str(e)}")
        return False

def test_full_workflow():
    """Test the complete research workflow with NASA integration."""
    console.print("\n🚀 [bold blue]Testing Complete Research Workflow[/bold blue]")
    
    try:
        # Define a test research question
        test_question = "What are the effects of microgravity on plant growth and development?"
        
        console.print(f"\n📋 [yellow]Research Question:[/yellow] {test_question}")
        
        # Initialize state with proper structure
        initial_state = {
            "messages": [HumanMessage(content=test_question)],
            "retrieval_brief": "",
            "retriever_messages": [],
            "tool_call_iterations": 0,
            "compressed_notes": "",
            "raw_notes": [],
            "next_step": "scoping"
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(description="Running complete research workflow...", total=None)
            
            # Run the complete workflow
            final_result = master_research_agent.invoke(initial_state)
            
            progress.update(task, completed=True)
        
        console.print("\n✅ [green]Workflow Test Completed Successfully![/green]")
        
        # Display the final research report
        if final_result.get("compressed_notes"):
            console.print("\n📄 [bold blue]Generated Research Report:[/bold blue]")
            markdown_report = Markdown(final_result["compressed_notes"])
            console.print(Panel(markdown_report, 
                              title="Final Research Report", 
                              border_style="blue",
                              padding=1))
        
        return True
        
    except Exception as e:
        console.print(f"\n❌ [red]Workflow Test Failed:[/red] {str(e)}")
        console.print(f"[red]Error details:[/red] {type(e).__name__}")
        import traceback
        console.print(f"[red]Traceback:[/red] {traceback.format_exc()}")
        return False

def test_retrieval_agent_only():
    """Test just the retrieval agent with NASA integration."""
    console.print("\n🔬 [bold blue]Testing Retrieval Agent with NASA Integration[/bold blue]")
    
    try:
        from src.retrieval_agent import retrieval_agent
        from src.state_retrieval import RetrieverState
        
        test_question = "What are NASA's current Mars exploration missions?"
        
        console.print(f"\n📋 [yellow]Research Question:[/yellow] {test_question}")
        
        # Initialize retrieval state
        initial_state = {
            "retriever_messages": [HumanMessage(content=test_question)],
            "tool_call_iterations": 0,
            "retrieval_topic": test_question,
            "compressed_notes": "",
            "raw_notes": []
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(description="Running retrieval agent...", total=None)
            
            # Run the retrieval agent
            result = retrieval_agent.invoke(initial_state)
            
            progress.update(task, completed=True)
        
        console.print("\n✅ [green]Retrieval Agent Test Completed![/green]")
        
        # Display the compressed notes
        if result.get("compressed_notes"):
            console.print("\n📄 [bold blue]Research Findings:[/bold blue]")
            console.print(Panel(result["compressed_notes"][:1000] + "..." if len(result["compressed_notes"]) > 1000 else result["compressed_notes"], 
                              title="Retrieval Agent Output", 
                              border_style="blue"))
        
        return True
        
    except Exception as e:
        console.print(f"\n❌ [red]Retrieval Agent Test Failed:[/red] {str(e)}")
        console.print(f"[red]Error details:[/red] {type(e).__name__}")
        return False

def run_all_tests():
    """Run all tests in sequence."""
    console.print(Panel.fit(
        "[bold blue]🧪 Space Biology Knowledge Engine Test Suite[/bold blue]\n"
        "[dim]Testing NASA Web Search integration and complete workflow[/dim]",
        border_style="blue"
    ))
    
    # Check environment setup
    console.print("\n🔧 [bold blue]Checking Environment Setup[/bold blue]")
    
    required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        console.print(f"❌ [red]Missing environment variables:[/red] {', '.join(missing_keys)}")
        console.print("[yellow]Please check your .env file[/yellow]")
        return False
    
    console.print("✅ [green]Environment variables configured[/green]")
    
    # Run individual tests
    tests = [
        ("NASA Web Search", test_nasa_search_integration),
        ("Database Search", test_database_search),
        ("Retrieval Agent", test_retrieval_agent_only),
        ("Complete Workflow", test_full_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        console.print(f"\n{'='*60}")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    console.print(f"\n{'='*60}")
    console.print("\n📊 [bold blue]Test Results Summary[/bold blue]")
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        color = "green" if passed else "red"
        console.print(f"[{color}]{status}[/{color}] {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        console.print(Panel.fit(
            "[bold green]🎉 ALL TESTS PASSED![/bold green]\n"
            "[dim]Your NASA web search integration is working perfectly![/dim]",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]❌ SOME TESTS FAILED[/bold red]\n"
            "[dim]Please check the error messages above[/dim]",
            border_style="red"
        ))
    
    return all_passed

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Tests interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n\n[red]Unexpected error during testing:[/red] {str(e)}")
        sys.exit(1)