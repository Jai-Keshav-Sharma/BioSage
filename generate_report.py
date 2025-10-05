#!/usr/bin/env python3
"""
Test file that generates a final markdown-formatted research report 
from the Space Biology Knowledge Engine workflow.

This test invokes the complete master research workflow and saves 
the final markdown report to a file for easy viewing.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def generate_research_report():
    """Generate a complete research report using the master workflow."""
    
    print("🚀 Space Biology Knowledge Engine - Report Generator")
    print("=" * 60)
    
    try:
        # Import the master workflow
        from src.master_research_workflow import master_research_agent
        from langchain_core.messages import HumanMessage
        
        # Use a new research query - different from previous tests
        research_question = (
            "What are the effects of cosmic radiation on DNA repair mechanisms "
            "in human cells during deep space exploration missions?"
        )
        
        print(f"🔬 Research Question:")
        print(f"   {research_question}")
        print("\n⏳ Executing complete research workflow...")
        print("   This may take several minutes as it searches both:")
        print("   • Local research paper database")  
        print("   • Official NASA websites")
        print("   • Processes and synthesizes findings")
        print("-" * 60)
        
        # Execute the complete master research workflow
        result = master_research_agent.invoke({
            "messages": [HumanMessage(content=research_question)],
            "next_step": "scoping",
            "tool_call_iterations": 0, 
            "compressed_notes": "",
            "raw_notes": [],
            "retriever_messages": []
        })
        
        # Extract the final markdown-formatted response
        if not result.get("messages") or len(result["messages"]) == 0:
            raise ValueError("Workflow completed but no messages were generated")
            
        final_response = result["messages"][-1].content
        
        if not final_response:
            raise ValueError("Workflow completed but final message has no content")
        
        print("✅ Research workflow completed successfully!")
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"research_report_{timestamp}.md"
        output_path = Path(output_file)
        
        # Save the markdown report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Space Biology Research Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Research Question:** {research_question}\n\n")
            f.write("---\n\n")
            f.write(final_response)
            
        print(f"\n📄 Report saved to: {output_path.absolute()}")
        print(f"📊 Report length: {len(final_response):,} characters")
        
        # Display a preview of the report
        print(f"\n📋 Report Preview:")
        print("-" * 60)
        
        # Show first 1000 characters as preview
        preview = final_response[:1000]
        if len(final_response) > 1000:
            preview += "\n\n[... report continues ...]"
            
        print(preview)
        print("-" * 60)
        
        # Show some statistics about the workflow result
        print(f"\n📈 Workflow Statistics:")
        print(f"   • Total messages in conversation: {len(result['messages'])}")
        print(f"   • Tool call iterations: {result.get('tool_call_iterations', 'N/A')}")
        print(f"   • Compressed notes length: {len(result.get('compressed_notes', '')):,} chars")
        print(f"   • Raw notes collected: {len(result.get('raw_notes', []))}")
        
        return True, output_path
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        import traceback
        print(f"Full error details:\n{traceback.format_exc()}")
        return False, None

def main():
    """Main function to run the report generation."""
    
    # Check environment variables
    required_vars = ["GROQ_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please ensure your .env file contains all required API keys.")
        return False
        
    print("✅ Environment variables configured")
    
    # Generate the research report
    success, output_file = generate_research_report()
    
    if success:
        print(f"\n🎉 SUCCESS! Research report generated successfully.")
        print(f"📁 Open the file to view the complete markdown report:")
        print(f"   {output_file}")
        print(f"\n💡 You can also view it in VS Code or any markdown viewer.")
    else:
        print(f"\n❌ FAILED! Could not generate research report.")
        return False
        
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Report generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {str(e)}")
        sys.exit(1)