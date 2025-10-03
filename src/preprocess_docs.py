
import fitz
from pathlib import Path
from typing import Dict, List 
import re
from tqdm import tqdm
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.docstore.document import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv


from src.config import (
    DOCUMENTS_PATH,
    QDRANT_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE
)



load_dotenv(override=True)


# PDF Extraction with Rich Metadata
def extract_text_from_pdf(pdf_path: Path) -> Dict:
    """
    Extract text and metadata from research paper PDFs
    Extracts: PMC ID, title, sections, page numbers
    """

    doc = fitz.open(pdf_path)

    # Extract PMC ID and title from filename
    filename = pdf_path.stem
    pmc_id = filename.split("_")[0] if "_" in filename else filename
    title = filename.split("_", 1)[1] if "_" in filename else filename

    # Try to extract the PDF metadata
    pdf_metadata = doc.metadata

    # Extract text page by page
    full_text = []
    pages_data = []

    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")

        # Clean up PDF artifacts 
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        text = re.sub(r'(?<=[a-z])-\n(?=[a-z])', '', text)  # Fix hyphenation
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces and tabs to single space, preserving newlines

        full_text.append(text)
        pages_data.append({
            "page_num": page_num, 
            "text": text
        })

    combined_text = "\n\n".join(full_text)

    # Detect organism/subject from title (useful for filtering later)
    organisms = []
    organisms_list = [
        'Arabidopsis', 'Burkholderia', 'Brassica', 'Penicillium',
        'Cohnella', 'Pseudomonas', 'Caenorhabditis', 'elegans',
        'human', 'mouse', 'rat'
    ]
    for organism in organisms_list:
        if organism.lower() in title.lower() or organism.lower() in combined_text[:2000].lower():
            organisms.append(organism)

    # Detect research type from title
    research_types = []
    if any(word in title.lower() for word in ['genomic', 'genome', 'gene']):
        research_types.append('genomic')
    if any(word in title.lower() for word in ['phenotypic', 'morphology']):
        research_types.append('phenotypic')
    if any(word in title.lower() for word in ['meta-analysis', 'review']):
        research_types.append('meta-analysis')
    if any(word in title.lower() for word in ['biofilm', 'microbial']):
        research_types.append('microbiology')

    return {
        'text': combined_text, 
        'pages_data': pages_data, 
        'metadata': {
            'source': str(pdf_path), 
            'pmc_id': pmc_id, 
            'title': title, 
            'filename': pdf_path.stem,
            'total_pages': len(doc), 
            'organisms': organisms if organisms else ['unknown'], 
            'research_types': research_types if research_types else ['general'], 
            'pdf_author': pdf_metadata.get('author', ''), 
            'pdf_subject': pdf_metadata.get('subject', ''),
        }
    }


# Load all dorcuments 
def load_all_documents() -> List[Document]:
    """Load and process all PDFs from the documents folder."""

    documents = []
    pdf_files = sorted(list(DOCUMENTS_PATH.glob("*.pdf")))

    print(f"\n📄 Found {len(pdf_files)} PDF files\n")
    print("-" * 80)

    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"[{i:2d}/{len(pdf_files)}] Processing: {pdf_path.name[:65]}...")
            extracted = extract_text_from_pdf(pdf_path)

            # Create Langchain Document 
            doc = Document(
                page_content=extracted['text'],
                metadata = extracted['metadata']
            )
            documents.append(doc)

            # Show extracted Metadata
            organisms = extracted['metadata']['organisms']
            types = extracted['metadata']['research_types']
            print(f"        └─ Organisms: {', '.join(organisms[:3])}")
            print(f"        └─ Types: {', '.join(types)}")

        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")

    print("-" * 80)
    print(f"\n✅ Successfully loaded {len(documents)} documents\n")
    return documents




# Chunk documents with page tracking
def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split document into chunks while preserving metadata
    Adds chunk_id and estimates page numbers
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP, 
        length_function=len, 
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    # Enrich chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i

        # Estimate which section (beginning, middle, or end of paper)
        doc_length = len(chunk.page_content)
        if i < len(chunks) * 0.2:
            chunk.metadata['section_position'] = 'introduction'
        elif i > len(chunks) * 0.8:
            chunk.metadata['section_position'] = 'conclusion'
        else:
            chunk.metadata['section_position'] = 'body'

    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks
