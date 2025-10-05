"""
NASA Space Biology RAG - Streamlit Frontend
Interactive interface for querying space biology research papers
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
from RAG.advanced_rag import AdvancedRAG

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
    page_title="NASA Space Biology RAG",
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
    
    /* Filter Pills */
    .filter-pill {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Stats Badge */
    .stats-badge {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* Spinner Overlay */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 600;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
    st.session_state.initialized = False


@st.cache_resource
def initialize_rag():
    """Initialize the RAG pipeline (cached)."""
    with st.spinner("🚀 Initializing RAG pipeline..."):
        rag = AdvancedRAG(k=8)
        rag.initialize()
    return rag


def main():
    # Animated Header with Icon
    st.markdown('''
        <div style="text-align: center; padding: 1rem 0;">
            <span style="font-size: 4rem;">🚀</span>
        </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">NASA Space Biology RAG</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subheader">✨ Explore space biology research with AI-powered semantic search ✨</p>',
        unsafe_allow_html=True
    )
    
    # Initialize RAG
    if not st.session_state.initialized:
        st.session_state.rag = initialize_rag()
        st.session_state.initialized = True
    
    # Sidebar - Modern Filters
    with st.sidebar:
        st.markdown("### ⚙️ Search Filters")
        st.markdown("---")
        
        # Organism filter with emoji
        st.markdown("#### 🧬 Organisms")
        organisms_options = [
            "Arabidopsis", "Mouse", "Human", "Rat", "C. elegans",
            "Drosophila", "E. coli", "Bacteria", "Fungi", "Yeast"
        ]
        organisms = st.multiselect(
            "Select organisms",
            options=organisms_options,
            help="Filter by specific organisms studied",
            label_visibility="collapsed"
        )
        
        st.markdown("")  # Spacing
        
        # Research type filter with emoji
        st.markdown("#### 🔬 Research Types")
        research_type_options = [
            "spaceflight", "microgravity", "genomic", "transcriptomic",
            "metabolomic", "phenotypic", "cellular", "molecular"
        ]
        research_types = st.multiselect(
            "Select research types",
            options=research_type_options,
            help="Filter by research methodology",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Settings Section
        st.markdown("### � Query Settings")
        
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
        
        st.markdown("---")
        
        # Info Section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Vector Database**  
            113,355+ documents indexed
            
            **Model**  
            Groq llama-3.3-70b-versatile
            
            **Embeddings**  
            CLIP multimodal
            """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            '<p style="text-align: center; color: #9ca3af; font-size: 0.85rem;">🌌 Built for NASA Space Apps 2025</p>',
            unsafe_allow_html=True
        )
    
    # Main content area with enhanced design
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
            if st.button("� Clear", use_container_width=True):
                st.rerun()
        
        if search_button and user_query:
            with st.spinner("� Analyzing research papers..."):
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
        ### 🚀 NASA Space Biology RAG
        
        This application uses **Retrieval-Augmented Generation (RAG)** to help you explore 
        NASA's space biology research papers from PubMed Central.
        
        #### 🔬 Features
        
        - **Semantic Search**: Find relevant research using natural language queries
        - **Metadata Filtering**: Filter by organisms, research types, and content modality
        - **Multimodal Retrieval**: Search across both text and images using CLIP embeddings
        - **Comparative Analysis**: Compare findings across different organisms
        - **AI-Powered Answers**: Get comprehensive answers powered by LLMs
        
        #### 🛠️ Technology Stack
        
        - **Frontend**: Streamlit
        - **Backend**: Python, LangChain
        - **Vector Database**: Qdrant
        - **Embeddings**: CLIP (OpenAI)
        - **LLM**: Groq (llama-3.3-70b-versatile)
        
        #### 📊 Dataset
        
        The knowledge base contains scientific papers covering:
        - Plant biology in space
        - Microbial genomics
        - Human health and physiology
        - Molecular and cellular responses to microgravity
        - And much more!
        
        #### 💡 Tips for Best Results
        
        1. Use specific, detailed questions
        2. Apply filters to narrow down results
        3. Try different prompt types for different analysis styles
        4. Use comparative analysis to understand cross-organism patterns
        
        ---
        
        **Built for NASA Space Apps Challenge** 🌌
        """)


if __name__ == "__main__":
    main()
