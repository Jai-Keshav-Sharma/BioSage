"""
Advanced RAG Pipeline with Metadata Filtering
Supports filtering by organisms, research types, and multimodal retrieval
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict
from functools import lru_cache

# Add parent directory to sys.path to allow importing from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from qdrant_client import models

from src.query_vectorstore import load_vectorstore


class AdvancedRAG:
    """
    Advanced RAG pipeline with filtering and multimodal support.
    """
    
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        k: int = 8
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name: Groq model to use
            temperature: LLM temperature (0 = deterministic)
            k: Number of documents to retrieve
        """
        self.model_name = model_name
        self.temperature = temperature
        self.k = k
        self.vector_store = None
        self.llm = None
        
        # Auto-initialize to avoid NoneType errors
        self.initialize()
        
    def initialize(self):
        """Load vector store and LLM."""
        print("⚙️  Initializing Advanced RAG pipeline...")
        
        # Load CLIP-based vector store
        self.vector_store = load_vectorstore()
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=2048,
            timeout=60
        )
        
        print("✅ Pipeline ready!\n")
    
    def create_filtered_retriever(
        self,
        organisms: Optional[List[str]] = None,
        research_types: Optional[List[str]] = None,
        modality: Optional[str] = None,
        k: Optional[int] = None
    ):
        """
        Create a retriever with metadata filters.
        
        Args:
            organisms: Filter by organisms (e.g., ['Arabidopsis', 'Mouse'])
            research_types: Filter by research types (e.g., ['spaceflight', 'genomic'])
            modality: Filter by type ('text' or 'image')
            k: Number of results (overrides default)
        
        Returns:
            Configured retriever
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first or create AdvancedRAG with auto-initialization.")
        
        # Get the requested number of results
        num_results = k or self.k
        
        # If we have filters, retrieve more results and filter in Python
        # This avoids needing Qdrant indexes
        has_filters = any([organisms, research_types, modality])
        
        if has_filters:
            # Retrieve 3x more results to ensure we have enough after filtering
            search_kwargs = {"k": num_results * 3}
        else:
            search_kwargs = {"k": num_results}
        
        # Store filter criteria for post-filtering
        self._current_filters = {
            "organisms": organisms,
            "research_types": research_types,
            "modality": modality,
            "target_k": num_results
        }
        
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    @staticmethod
    @lru_cache(maxsize=3)
    def create_prompt(context_type: str = "general") -> ChatPromptTemplate:
        """
        Create context-specific prompt templates (cached for performance).
        
        Args:
            context_type: Type of analysis ('general', 'comparative', 'mechanism')
        """
        PROMPTS = {
            "general": """You are an expert in NASA space biology research.

Context from research papers:
{context}

Question: {input}

Provide a detailed, scientifically accurate answer based on the context. Include specific examples and findings from the papers. 

IMPORTANT: Do NOT add a "References:" section at the end. Do NOT include citation numbers like [1], [2], etc. Just provide the answer directly.""",
            
            "comparative": """You are an expert in comparative space biology research.

Context from research papers:
{context}

Question: {input}

Compare and contrast findings across different studies. Highlight similarities, differences, and potential explanations. 

IMPORTANT: Do NOT add a "References:" section at the end. Do NOT include citation numbers like [1], [2], etc. Just provide the answer directly.""",
            
            "mechanism": """You are an expert in molecular and cellular mechanisms in space biology.

Context from research papers:
{context}

Question: {input}

Explain the underlying mechanisms, molecular pathways, and cellular processes involved. Mention specific genes, proteins, or signaling pathways. 

IMPORTANT: Do NOT add a "References:" section at the end. Do NOT include citation numbers like [1], [2], etc. Just provide the answer directly."""
        }
        
        template = PROMPTS.get(context_type, PROMPTS["general"])
        return ChatPromptTemplate.from_template(template)
    
    def _filter_documents(self, documents: List[Document]) -> List[Document]:
        """
        Filter documents based on metadata criteria stored in _current_filters.
        This allows filtering without requiring Qdrant indexes.
        
        Args:
            documents: List of documents from retriever
            
        Returns:
            Filtered list of documents (up to target_k)
        """
        if not hasattr(self, '_current_filters'):
            return documents
        
        filters = self._current_filters
        organisms = filters.get("organisms")
        research_types = filters.get("research_types")
        modality = filters.get("modality")
        target_k = filters.get("target_k", self.k)
        
        filtered_docs = []
        
        for doc in documents:
            metadata = doc.metadata
            
            # Check organism filter
            if organisms:
                doc_organisms = metadata.get("organisms", [])
                if not any(org in doc_organisms for org in organisms):
                    continue
            
            # Check research_types filter
            if research_types:
                doc_research_types = metadata.get("research_types", [])
                if not any(rt in doc_research_types for rt in research_types):
                    continue
            
            # Check modality filter
            if modality:
                doc_type = metadata.get("type")
                if doc_type != modality:
                    continue
            
            filtered_docs.append(doc)
            
            # Stop once we have enough results
            if len(filtered_docs) >= target_k:
                break
        
        return filtered_docs
    
    def query(
        self,
        question: str,
        organisms: Optional[List[str]] = None,
        research_types: Optional[List[str]] = None,
        modality: Optional[str] = None,
        prompt_type: str = "general",
        k: Optional[int] = None,
        show_sources: bool = True
    ) -> Dict:
        """
        Query the RAG pipeline with optional filters.
        
        Args:
            question: User question
            organisms: Filter by organisms
            research_types: Filter by research types
            modality: Filter by modality ('text' or 'image')
            prompt_type: Prompt template to use
            k: Number of results
            show_sources: Display sources
        
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Create filtered retriever
            retriever = self.create_filtered_retriever(
                organisms=organisms,
                research_types=research_types,
                modality=modality,
                k=k
            )
            
            # Create prompt (cached)
            prompt = self.create_prompt(prompt_type)
            
            # Build and execute chain
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Execute query (no terminal output)
            result = retrieval_chain.invoke({"input": question})
            
            # Get raw context documents
            context_docs = result.get("context", [])
            
            # Apply post-filtering if we have filter criteria
            if hasattr(self, '_current_filters') and any([organisms, research_types, modality]):
                context_docs = self._filter_documents(context_docs)
                
                # If we don't have enough filtered results, warn but continue
                if len(context_docs) == 0:
                    print(f"⚠️  No documents matched the filters. Trying without filters...")
                    # Re-run without filters
                    retriever = self.vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": k or self.k}
                    )
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    result = retrieval_chain.invoke({"input": question})
                    context_docs = result.get("context", [])
            
            answer = result.get("answer", "No answer generated")
            
            # Clean up answer: remove References section
            answer = self._clean_answer(answer)
            
            # Display sources only if requested (for terminal usage)
            if show_sources and context_docs:
                self._display_sources(context_docs)
            
            return {
                "question": question,
                "answer": answer,
                "sources": context_docs,
                "filters": {
                    "organisms": organisms,
                    "research_types": research_types,
                    "modality": modality
                }
            }
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "sources": [],
                "filters": {}
            }
    
    def _display_sources(self, documents: List[Document]):
        """Display retrieved sources with metadata (optimized)."""
        print("\n📚 Sources")
        print("=" * 80)
        
        seen_sources = set()
        counts = {"text": 0, "image": 0}
        
        for doc in documents:
            metadata = doc.metadata
            pmc_id = metadata.get("pmc_id", "Unknown")
            
            if pmc_id in seen_sources:
                continue
            seen_sources.add(pmc_id)
            
            doc_type = metadata.get("type", "text")
            counts[doc_type] = counts.get(doc_type, 0) + 1
            
            icon = "📸" if doc_type == "image" else "📝"
            title = metadata.get("title", "Unknown")[:70]
            organisms = ", ".join(metadata.get("organisms", ["Unknown"])[:3])
            research_types = ", ".join(metadata.get("research_types", ["Unknown"])[:2])
            
            print(f"{icon} [{pmc_id}] {title}")
            print(f"    {organisms} | {research_types}")
        
        print(f"\n📊 Retrieved: {counts['text']} text sources, {counts.get('image', 0)} images")
    
    @staticmethod
    def _clean_answer(answer: str) -> str:
        """
        Remove References section and citation markers from the answer.
        
        Args:
            answer: Raw answer from LLM
            
        Returns:
            Cleaned answer without references
        """
        import re
        
        # Remove everything after "References:" section (most important)
        # This catches: "References:", "References :", "References", etc.
        if 'References:' in answer or 'references:' in answer.lower():
            # Split on the References section
            parts = re.split(r'\n+\s*References?\s*:.*', answer, maxsplit=1, flags=re.IGNORECASE | re.DOTALL)
            answer = parts[0] if parts else answer
        
        # Remove citation markers like [1], [2], [35], etc.
        answer = re.sub(r'\s*\[\d+\]\s*', ' ', answer)
        
        # Remove inline citations in format: (Author et al., YEAR)
        # But be careful to only remove when it's clearly a citation
        answer = re.sub(r'\s*\([A-Z][a-z]+\s+et\s+al\.\s*,?\s*\d{4}\)\s*', ' ', answer)
        
        # Remove inline citations: (Author, YEAR)
        answer = re.sub(r'\s*\([A-Z][a-z]+\s*,?\s*\d{4}\)\s*', ' ', answer)
        
        # Clean up phrases like "According to Author et al. (YEAR) and Author2 et al. (YEAR),"
        # Replace with just "Studies show that" or similar
        answer = re.sub(
            r'According to\s+(?:[A-Z][a-z]+\s+et\s+al\.\s*\(\d{4}\)\s*(?:and|,)?\s*)+',
            'Studies show that ',
            answer,
            flags=re.IGNORECASE
        )
        
        # Remove standalone "et al. (YEAR)" that might be left
        answer = re.sub(r'\s+et\s+al\.\s*\(\d{4}\)\s*', ' ', answer)
        
        # Clean up multiple spaces
        answer = re.sub(r' {2,}', ' ', answer)
        
        # Clean up spaces before punctuation
        answer = re.sub(r'\s+([.,;!?])', r'\1', answer)
        
        # Clean up multiple newlines
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        # Remove trailing/leading whitespace
        return answer.strip()
    
    def compare_organisms(
        self,
        question: str,
        organisms: List[str],
        research_types: Optional[List[str]] = None,
        k: int = 10
    ) -> Dict:
        """
        Compare findings across different organisms.
        
        Args:
            question: Research question
            organisms: List of organisms to compare
            research_types: Optional filter for research types
            k: Results per organism
        """
        if not organisms:
            raise ValueError("At least one organism must be specified")
        
        results = {}
        individual_findings = {}
        
        # Query each organism
        for organism in organisms:
            try:
                result = self.query(
                    question=f"{question} in {organism}",
                    organisms=[organism],
                    research_types=research_types,
                    prompt_type="comparative",
                    k=k,
                    show_sources=False
                )
                results[organism] = result
                # Extract answer text for display
                individual_findings[organism] = result.get('answer', '')
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                results[organism] = {"answer": error_msg, "sources": []}
                individual_findings[organism] = error_msg
        
        # Build synthesis prompt with truncated answers
        findings = "\n\n".join([
            f"{org}:\n{res['answer'][:300]}..."
            for org, res in results.items()
            if "Error" not in res['answer']
        ])
        
        synthesis_prompt = f"""Based on the following findings for different organisms, provide a comparative summary:

{findings}

Highlight key similarities and differences across organisms. 

IMPORTANT: Do NOT add a "References:" section. Do NOT include citation numbers like [1], [2], etc. Just provide the comparative analysis directly."""
        
        try:
            synthesis = self.llm.invoke(synthesis_prompt)
            synthesis_content = self._clean_answer(synthesis.content)
        except Exception as e:
            synthesis_content = "Error generating comparative summary"
        
        return {
            "question": question,
            "organisms": organisms,
            "individual_results": results,
            "individual_findings": individual_findings,
            "synthesis": synthesis_content
        }


def main():
    """Demo of advanced RAG features."""
    load_dotenv(override=True)
    
    print("=" * 80)
    print("🧬 Advanced NASA Space Biology RAG")
    print("=" * 80)
    
    # Initialize
    rag = AdvancedRAG(k=8)
    rag.initialize()
    
    # Example 1: Basic query
    print("\n" + "=" * 80)
    print("Example 1: Basic Query")
    print("=" * 80)
    rag.query(
        "What are the effects of microgravity on gene expression?",
        show_sources=True
    )
    
    # Example 2: Filtered by organism
    print("\n\n" + "=" * 80)
    print("Example 2: Organism-Specific Query")
    print("=" * 80)
    rag.query(
        "How does spaceflight affect growth?",
        organisms=["Arabidopsis"],
        research_types=["spaceflight"],
        show_sources=True
    )
    
    # Example 3: Comparative analysis
    print("\n\n" + "=" * 80)
    print("Example 3: Comparative Analysis")
    print("=" * 80)
    rag.compare_organisms(
        "What are the cellular stress responses to spaceflight?",
        organisms=["Arabidopsis", "Mouse", "Human"],
        k=5
    )
    
    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
