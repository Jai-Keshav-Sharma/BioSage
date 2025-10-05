"""
NASA Space Biology RAG - Enhanced Streamlit Frontend
Interactive interface with both Quick RAG and Deep Research Workflow
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from RAG.advanced_rag import AdvancedRAG
from src.master_research_workflow import master_research_agent

# Load environment variables
load_dotenv(override=True)


def _display_sources(sources):
    """
    Display source documents in modern cards with proper formatting.
    """
    seen_sources = set()
    text_count = 0
    image_count = 0
    
    for doc in sources:
        metadata = doc.metadata
        pmc_id = metadata.get("pmc_id", "Unknown")
        doc_type = metadata.get("type", "text")
        
        # Skip duplicate text sources (but show all images)
        if doc_type != "image":
            if pmc_id in seen_sources:
                continue
            seen_sources.add(pmc_id)
        
        title = metadata.get("title", "Unknown")
        organisms_list = metadata.get("organisms", ["Unknown"])
        research_types_list = metadata.get("research_types", ["Unknown"])
        
        # Count types
        if doc_type == "image":
            image_count += 1
            icon = "📸"
            type_label = "Image Source"
        else:
            text_count += 1
            icon = "📝"
            type_label = "Text Source"
        
        # Display source card with modern styling
        if doc_type == "image":
            st.markdown(f"""
            <div class="source-card">
                <h4>{icon} {type_label}: {title}</h4>
                <p><strong>PMC ID:</strong> {pmc_id}</p>
                <p><strong>Page:</strong> {metadata.get('page', 'N/A')}</p>
                <p><strong>Image ID:</strong> {metadata.get('image_id', 'N/A')}</p>
                <p><strong>Organisms:</strong> {', '.join(organisms_list[:5])}</p>
                <p><strong>Research Types:</strong> {', '.join(research_types_list[:3])}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="source-card">
                <h4>{icon} {title}</h4>
                <p><strong>PMC ID:</strong> {pmc_id}</p>
                <p><strong>Organisms:</strong> {', '.join(organisms_list[:5])}</p>
                <p><strong>Research Types:</strong> {', '.join(research_types_list[:3])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary
    if text_count > 0 or image_count > 0:
        st.info(f"📊 Retrieved: {text_count} text sources, {image_count} images")


# Page configuration
st.set_page_config(
    page_title="NASA Space Biology Research Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header with Animated Gradient */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 2rem 0 1rem 0;
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
        letter-spacing: -1px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Subheader */
    .subheader {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #ffffff 100%);
    }
    
    /* Enhanced Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Source Cards with Modern Design */
    .source-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border-left: 5px solid #667eea;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.15);
    }
    
    .source-card h4 {
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .source-card p {
        color: #4b5563;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    
    .source-card strong {
        color: #374151;
        font-weight: 600;
    }
    
    /* Chat Message Styling - ChatGPT-like */
    .chat-message {
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        max-width: 100%;
        font-family: 'Inter', sans-serif;
    }
    
    .user-message {
        background: #f7f7f8;
        border: 1px solid #e5e5e6;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: #ffffff;
        border: 1px solid #e5e5e6;
        margin-right: 20%;
    }
    
    /* Chat input styling */
    .stForm {
        border: none !important;
        background: transparent !important;
    }
    
    .stForm > div {
        border: none !important;
    }
    
    /* Chat container */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background: #fafafa;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Research Mode Pills */
    .mode-pill {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .mode-rag {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
    }
    
    .mode-deep {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 1.1rem;
        color: #6b7280;
        padding: 1rem 0;
        background-color: transparent;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea;
        border-bottom: 3px solid #667eea;
    }
    
    /* Input Fields */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e5e7eb;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Multiselect */
    .stMultiSelect [data-baseweb="select"] {
        border-radius: 12px;
    }
    
    /* Success/Info/Warning Messages */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-weight: 500;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #374151;
        border-radius: 8px;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    }
    
    /* Markdown content styling */
    .markdown-content {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        line-height: 1.8;
    }
    
    .markdown-content h1, .markdown-content h2, .markdown-content h3 {
        color: #1f2937;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .markdown-content h1 {
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .markdown-content h2 {
        border-left: 4px solid #10b981;
        padding-left: 1rem;
    }
    
    .markdown-content ul, .markdown-content ol {
        padding-left: 2rem;
    }
    
    .markdown-content li {
        margin: 0.5rem 0;
    }
    
    .markdown-content blockquote {
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        background: #fef3c7;
        border-radius: 8px;
    }
    
    .markdown-content code {
        background: #f3f4f6;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
    }
    
    .markdown-content pre {
        background: #1f2937;
        color: #f9fafb;
        padding: 1rem;
        border-radius: 8px;
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
    st.session_state.initialized = False

# Simple deep research mode - no session state needed


@st.cache_resource
def initialize_rag():
    """Initialize the RAG pipeline (cached)."""
    with st.spinner("🚀 Initializing RAG pipeline..."):
        rag = AdvancedRAG(k=8)
        rag.initialize()
    return rag


def run_deep_research_workflow(user_query: str) -> str:
    """
    Execute the simple research workflow and return final report.
    
    Args:
        user_query: User's research question
        
    Returns:
        Final markdown research report
    """
    try:
        # Simple unidirectional execution
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "compressed_notes": "",
            "raw_notes": [],
            "retriever_messages": [],
            "tool_call_iterations": 0
        }
        
        # Execute the workflow
        result = master_research_agent.invoke(initial_state)
        
        # Extract the final response
        messages = result.get("messages", [])
        if messages:
            return messages[-1].content
        else:
            return "❌ No response generated from the workflow."
        
    except Exception as e:
        return f"❌ Deep research error: {str(e)}"



def main():
    # Animated Header with Icon
    st.markdown('''
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 4rem;">🚀</span>
        </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">NASA Space Biology Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subheader">✨ Explore space biology research with AI-powered search and deep analysis ✨</p>',
        unsafe_allow_html=True
    )
    
    # Initialize RAG
    if not st.session_state.initialized:
        st.session_state.rag = initialize_rag()
        st.session_state.initialized = True
    
    # Sidebar - Research Mode Selection and Filters
    with st.sidebar:
        st.markdown("### 🎯 Research Mode")
        
        research_mode = st.selectbox(
            "Choose Analysis Type",
            options=["quick_rag", "deep_research"],
            format_func=lambda x: {
                "quick_rag": "🔍 Quick Search (RAG)",
                "deep_research": "🧠 Deep Research (LangGraph)"
            }[x],
            help="Quick Search for fast answers, Deep Research for comprehensive analysis",
            key="research_mode"
        )
        
        # Display current mode with styled pill
        if research_mode == "quick_rag":
            st.markdown('<span class="mode-pill mode-rag">🔍 Quick Search Mode</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="mode-pill mode-deep">🧠 Deep Research Mode</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Define options for both modes
        organisms_options = [
            "Arabidopsis", "Mouse", "Human", "Rat", "C. elegans",
            "Drosophila", "E. coli", "Bacteria", "Fungi", "Yeast"
        ]
        research_type_options = [
            "spaceflight", "microgravity", "genomic", "transcriptomic",
            "metabolomic", "phenotypic", "cellular", "molecular"
        ]
        
        # Filters (only for RAG mode)
        if research_mode == "quick_rag":
            st.markdown("### ⚙️ Search Filters")
            
            # Organism filter with emoji
            st.markdown("#### 🧬 Organisms")
            organisms = st.multiselect(
                "Select organisms",
                options=organisms_options,
                help="Filter by specific organisms studied",
                label_visibility="collapsed"
            )
            
            st.markdown("")  # Spacing
            
            # Research type filter with emoji
            st.markdown("#### 🔬 Research Types")
            research_types = st.multiselect(
                "Select research types",
                options=research_type_options,
                help="Filter by research methodology",
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Settings Section
            st.markdown("### ⚙️ Query Settings")
            
            prompt_type = st.selectbox(
                "Analysis Style",
                options=["general", "comparative", "mechanism"],
                format_func=lambda x: {
                    "general": "📊 General Analysis",
                    "comparative": "🔄 Comparative Study", 
                    "mechanism": "⚙️ Mechanism Deep-Dive"
                }[x],
                help="Choose the type of analysis"
            )
            
            num_results = st.slider(
                "📚 Documents to Retrieve",
                min_value=3,
                max_value=20,
                value=8,
                help="Number of documents to retrieve"
            )
            
            show_sources = st.checkbox(
                "📖 Show Detailed Sources",
                value=True,
                help="Display detailed source information"
            )
        
        else:  # Deep research mode
            # Initialize default values for variables used in other tabs
            organisms = []
            research_types = []
            prompt_type = "general"
            num_results = 8
            show_sources = True
            
            st.markdown("### 🧠 Deep Research Settings")
            st.info("""
            **Deep Research Mode Features:**
            - Multi-step analysis workflow
            - Comprehensive literature review
            - Structured research reports
            - Evidence synthesis
            - Multi-turn conversations
            """)
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation", help="Start a new research session"):
                st.session_state.deep_research_messages = []
                st.session_state.deep_research_state = None
                st.rerun()
        
        st.markdown("---")
        
        # Info Section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Vector Database**  
            113,355+ documents indexed
            
            **Models**  
            - RAG: Groq llama-3.3-70b-versatile
            - Deep Research: Groq llama-3.3-70b-versatile
            
            **Embeddings**  
            CLIP multimodal
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            '<p style="text-align: center; color: #9ca3af; font-size: 0.85rem;">🌌 Built for NASA Space Apps 2025</p>',
            unsafe_allow_html=True
        )
    
    # Main content area based on selected mode
    if research_mode == "quick_rag":
        # Quick RAG Mode - Enhanced existing functionality
        tab1, tab2, tab3 = st.tabs(["🔍 Search", "🧬 Compare", "📚 About"])
        
        # Tab 1: Single Query with Modern Design
        with tab1:
            # Hero section
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                            padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
                    <h2 style="margin: 0; color: #1f2937;">💬 Ask Your Question</h2>
                    <p style="color: #6b7280; margin-top: 0.5rem;">
                        Explore cutting-edge space biology research with natural language queries
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Example questions
            with st.expander("💡 Example Questions"):
                st.markdown("""
                - How does microgravity affect gene expression in plants?
                - What are the effects of spaceflight on the immune system?
                - How do organisms adapt to the space environment?
                - What are the molecular mechanisms of muscle atrophy in space?
                - How does radiation exposure affect DNA repair mechanisms?
                """)
            
            # Query input with modern styling
            user_query = st.text_area(
                "Your Question",
                placeholder="e.g., How does microgravity affect plant growth in space?",
                height=120,
                label_visibility="collapsed"
            )
            
            # Action buttons row
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search_button = st.button("🚀 Search", use_container_width=True, type="primary")
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.rerun()
            
            if search_button and user_query:
                with st.spinner("🔍 Analyzing research papers..."):
                    try:
                        # Execute query with filters
                        result = st.session_state.rag.query(
                            question=user_query,
                            organisms=organisms if organisms else None,
                            research_types=research_types if research_types else None,
                            modality=None,
                            prompt_type=prompt_type,
                            k=num_results,
                            show_sources=False
                        )
                        
                        # Display answer with enhanced styling
                        st.markdown("---")
                        st.markdown("""
                            <div style="background: linear-gradient(135deg, #10b98115 0%, #0ea5e915 100%); 
                                        padding: 2rem; border-radius: 16px; margin: 1.5rem 0;">
                                <h3 style="margin: 0 0 1rem 0; color: #065f46;">
                                    <span style="font-size: 1.5rem;">✨</span> Answer
                                </h3>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"<div style='color: #1f2937; line-height: 1.8;'>{result['answer']}</div>", 
                                   unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Display sources with modern cards
                        if result.get("sources") and show_sources:
                            st.markdown("---")
                            st.markdown("""
                                <h3 style="color: #1f2937; margin: 2rem 0 1rem 0;">
                                    📄 Source Documents
                                </h3>
                            """, unsafe_allow_html=True)
                            
                            _display_sources(result["sources"])
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            elif search_button:
                st.warning("⚠️ Please enter a question to search.")
        
        # Tab 2: Comparative Analysis with Enhanced Design
        with tab2:
            # Hero section
            st.markdown("""
                <div style="background: linear-gradient(135deg, #f59e0b15 0%, #ef444415 100%); 
                            padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
                    <h2 style="margin: 0; color: #1f2937;">🧬 Comparative Analysis</h2>
                    <p style="color: #6b7280; margin-top: 0.5rem;">
                        Compare biological responses across different organisms in space
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Query input
            comp_query = st.text_area(
                "Research Question",
                placeholder="e.g., What are the cellular stress responses to spaceflight?",
                height=100,
                label_visibility="collapsed",
                key="comp_query"
            )
            
            # Organism selection
            st.markdown("**Select organisms to compare** (minimum 2)")
            comp_organisms = st.multiselect(
                "Organisms",
                options=organisms_options,
                default=["Arabidopsis", "Mouse"],
                label_visibility="collapsed",
                key="comp_organisms"
            )
            
            # Action button
            compare_button = st.button("🔬 Compare", type="primary", use_container_width=True)
            
            if compare_button and comp_query and len(comp_organisms) >= 2:
                with st.spinner(f"🧬 Analyzing {len(comp_organisms)} organisms..."):
                    try:
                        result = st.session_state.rag.compare_organisms(
                            question=comp_query,
                            organisms=comp_organisms,
                            research_types=research_types if research_types else None,
                            k=num_results
                        )
                        
                        # Display comparative analysis
                        st.markdown("---")
                        st.markdown("""
                            <div style="background: linear-gradient(135deg, #ec489915 0%, #f97316 15 100%); 
                                        padding: 2rem; border-radius: 16px; margin: 1.5rem 0;">
                                <h3 style="margin: 0 0 1rem 0; color: #9a3412;">
                                    <span style="font-size: 1.5rem;">📊</span> Comparative Analysis
                                </h3>
                        """, unsafe_allow_html=True)
                        
                        # Individual organism findings
                        for organism, findings in result.get("individual_findings", {}).items():
                            with st.expander(f"🧬 {organism}", expanded=True):
                                st.markdown(findings)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Synthesis
                        if result.get("synthesis"):
                            st.markdown("---")
                            st.markdown("""
                                <div style="background: linear-gradient(135deg, #8b5cf615 0%, #a78bfa15 100%); 
                                            padding: 2rem; border-radius: 16px; margin: 1.5rem 0;">
                                    <h3 style="margin: 0 0 1rem 0; color: #5b21b6;">
                                        <span style="font-size: 1.5rem;">🔍</span> Key Insights
                                    </h3>
                            """, unsafe_allow_html=True)
                            st.markdown(result["synthesis"])
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            elif compare_button and len(comp_organisms) < 2:
                st.warning("⚠️ Please select at least 2 organisms to compare.")
            elif compare_button and not comp_query:
                st.warning("⚠️ Please enter a research question.")
        
        # Tab 3: About with Enhanced Design
        with tab3:
            # Hero section
            st.markdown("""
                <div style="background: linear-gradient(135deg, #3b82f615 0%, #2563eb15 100%); 
                            padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
                    <h2 style="margin: 0; color: #1f2937;">📚 About This Application</h2>
                    <p style="color: #6b7280; margin-top: 0.5rem;">
                        Advanced AI-powered research assistant for space biology
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 🚀 NASA Space Biology Research Assistant
            
            This application provides two powerful modes for exploring NASA's space biology research:
            
            #### 🔍 Quick Search (RAG) Mode
            
            - **Semantic Search**: Find relevant research using natural language queries
            - **Metadata Filtering**: Filter by organisms, research types, and content modality
            - **Multimodal Retrieval**: Search across both text and images using CLIP embeddings
            - **Comparative Analysis**: Compare findings across different organisms
            - **AI-Powered Answers**: Get comprehensive answers powered by LLMs
            
            #### 🧠 Deep Research (LangGraph) Mode
            
            - **Multi-Step Analysis**: Comprehensive research workflow with scoping, retrieval, and synthesis
            - **Structured Reports**: Well-formatted markdown reports with executive summaries
            - **Evidence Synthesis**: Advanced reasoning over multiple research papers
            - **Multi-Turn Conversations**: Continue research conversations with context awareness
            - **Advanced Orchestration**: LangGraph-powered workflow for thorough analysis
            
            #### 🛠️ Technology Stack
            
            - **Frontend**: Streamlit with custom CSS styling
            - **Backend**: Python, LangChain, LangGraph
            - **Vector Database**: Qdrant with CLIP embeddings
            - **Models**: 
              - RAG: Groq llama-3.3-70b-versatile
              - Deep Research: Groq llama-3.3-70b-versatile
            - **Orchestration**: LangGraph state management
            
            #### 📊 Dataset
            
            The knowledge base contains 113,355+ scientific papers covering:
            - Plant biology in space (Arabidopsis, growth responses)
            - Microbial genomics and biofilms
            - Human health and physiology in microgravity
            - Molecular and cellular responses to spaceflight
            - Radiation effects and DNA repair mechanisms
            - Bone and muscle adaptation to space environment
            
            #### 💡 Usage Tips
            
            **For Quick Search:**
            - Use specific, detailed questions
            - Apply filters to narrow down results
            - Try different prompt types for different analysis styles
            
            **For Deep Research:**
            - Ask complex, open-ended research questions
            - Allow time for multi-step analysis
            - Continue conversations to dive deeper into topics
            - Download reports for comprehensive documentation
            
            ---
            
            **Built for NASA Space Apps Challenge 2025** 🌌
            """)
    
    else:  # Deep Research Mode
        # Deep Research Mode - New LangGraph Integration
        st.markdown("""
            <div style="background: linear-gradient(135deg, #8b5cf615 0%, #a78bfa15 100%); 
                        padding: 2rem; border-radius: 16px; margin-bottom: 2rem;">
                <h2 style="margin: 0; color: #1f2937;">🧠 Deep Research Assistant</h2>
                <p style="color: #6b7280; margin-top: 0.5rem;">
                    Comprehensive multi-step research analysis with LangGraph workflow
                </p>
            </div>
        """, unsafe_allow_html=True)
        

        
        # Chat input section - ChatGPT style
        st.markdown("### 💭 Chat with Research Assistant")
        
        # Show examples
        with st.expander("💡 Deep Research Examples"):
            st.markdown("""
            **Complex Research Questions:**
            - What is the effect of microgravity on bones?
            - Provide a comprehensive analysis of microgravity effects on plant gene expression
            - What are the molecular mechanisms underlying bone loss in spaceflight?
            - How do different organisms adapt their stress response pathways in space?
            - Analyze the role of biofilms in space station microbial ecology
            - Compare cardiovascular deconditioning mechanisms across different species
            """)
        
        # Simple query input
        with st.form(key="research_form", clear_on_submit=True):
            deep_query = st.text_area(
                "Research Query",
                placeholder="Enter your space biology research question...",
                height=100,
                help="Ask complex research questions to get comprehensive analysis"
            )
            
            research_button = st.form_submit_button("� Generate Research Report", use_container_width=True, type="primary")
        
        # Process research query
        if research_button and deep_query:            
            # Execute deep research workflow
            with st.spinner("🧠 Conducting deep research analysis..."):
                try:
                    # Run the simplified workflow
                    research_report = run_deep_research_workflow(deep_query)
                    
                    # Display the research report
                    st.markdown("### 📋 Research Report")
                    st.markdown(research_report)
                    
                    # Download option
                    st.download_button(
                        label="📄 Download Report as Markdown",
                        data=research_report,
                        file_name=f"research_report_{deep_query[:30].replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Deep research error: {str(e)}")
                    st.info("💡 Try rephrasing your question or check that all dependencies are properly configured.")
        
        elif research_button:
            st.warning("⚠️ Please enter a research question to begin deep analysis.")
        
        # Information about the deep research workflow
        st.markdown("""
        ---
        
        ### 🎯 How Deep Research Works
        
        1. **� Research**: Searches relevant literature using advanced retrieval agents  
        2. **🧬 Synthesis**: Combines findings into a comprehensive markdown report
        
        **Features:**
        - Multi-step reasoning workflow
        - Comprehensive literature analysis
        - Structured markdown reports
        - Evidence-based synthesis
        """)


if __name__ == "__main__":
    main()