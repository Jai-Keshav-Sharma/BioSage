"""
Multimodal PDF Processing - Extract both text and images for CLIP embeddings
"""

import fitz
from pathlib import Path
from typing import Dict, List
import re
import io
import base64
from PIL import Image
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import (
    DOCUMENTS_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Pre-compiled regex patterns for performance
EXCESSIVE_NEWLINES = re.compile(r'\n{3,}')
HYPHENATION_PATTERN = re.compile(r'(?<=[a-z])-\n(?=[a-z])')
WHITESPACE_PATTERN = re.compile(r'[ \t]+')

# Organism and research type sets (from preprocess_docs.py)
ORGANISMS_SET = frozenset([
    # Plants
    'arabidopsis', 'thaliana', 'brassica',
    # Microorganisms (bacteria)
    'burkholderia', 'pseudomonas', 'bacillus', 'salmonella', 
    'escherichia', 'coli', 'staphylococcus', 'enterobacter',
    # Fungi
    'penicillium', 'aspergillus', 'candida', 'fusarium',
    # Model organisms
    'caenorhabditis', 'elegans', 'drosophila', 'melanogaster',
    # Mammals
    'human', 'mouse', 'rat', 'mice',
    # Other
    'tardigrade', 'yeast'
])

# Additional organism pattern for fallback (catches genus-species patterns)
ORGANISM_PATTERN = re.compile(r'\b([A-Z][a-z]+)\s+([a-z]+)\b')

GENOMIC_KEYWORDS = frozenset(['genomic', 'genome', 'gene', 'transcriptome', 'proteome', 'metabolome'])
PHENOTYPIC_KEYWORDS = frozenset(['phenotypic', 'morphology'])
META_ANALYSIS_KEYWORDS = frozenset(['meta-analysis', 'review'])
MICROBIOLOGY_KEYWORDS = frozenset(['biofilm', 'microbial', 'antimicrobial', 'resistance'])
SPACEFLIGHT_KEYWORDS = frozenset(['spaceflight', 'microgravity', 'space', 'radiation', 'ISS'])
TISSUE_KEYWORDS = frozenset(['bone', 'muscle', 'cardiovascular', 'neurological', 'brain', 'immune'])


def extract_multimodal_from_pdf(pdf_path: Path, extract_images: bool = True) -> Dict:
    """
    Extract both text and images from PDF for multimodal CLIP embeddings.
    Includes organism and research type detection.
    
    Args:
        pdf_path: Path to PDF file
        extract_images: Whether to extract images (default: True)
    
    Returns:
        Dictionary with text chunks and image data
    """
    doc = fitz.open(pdf_path)
    
    # Extract PMC ID and title from filename
    filename = pdf_path.stem
    pmc_id = filename.split("_")[0] if "_" in filename else filename
    title = filename.split("_", 1)[1] if "_" in filename else filename
    
    # Get PDF metadata
    pdf_metadata = doc.metadata
    
    # Storage for text and images
    text_documents = []
    image_documents = []
    full_text = []
    
    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Process each page
    for page_num, page in enumerate(doc, 1):
        # Extract and clean text
        text = page.get_text("text")
        if text.strip():
            # Clean up PDF artifacts
            text = HYPHENATION_PATTERN.sub('', text)
            text = EXCESSIVE_NEWLINES.sub('\n\n', text)
            text = WHITESPACE_PATTERN.sub(' ', text)
            
            full_text.append(text)
            
        # Extract images if enabled
        if extract_images:
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    
                    # Try to extract image with timeout protection
                    try:
                        base_image = doc.extract_image(xref)
                    except KeyboardInterrupt:
                        raise
                    except Exception as img_extract_err:
                        # Skip problematic images
                        continue
                        
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    
                    # Skip very small images (likely icons/decorations)
                    if pil_image.width < 50 or pil_image.height < 50:
                        continue
                    
                    # Create unique identifier
                    image_id = f"{pmc_id}_page_{page_num}_img_{img_index}"
                    
                    # Store image as base64 for later retrieval
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Create document for image
                    image_doc = Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={
                            "source": str(pdf_path),
                            "pmc_id": pmc_id,
                            "title": title,
                            "filename": pdf_path.stem,
                            "page": page_num,
                            "total_pages": len(doc),
                            "type": "image",
                            "image_id": image_id,
                            "image_data": img_base64,  # Store for retrieval
                            "image_bytes": image_bytes,  # For embedding
                            "image_size": f"{pil_image.width}x{pil_image.height}"
                        }
                    )
                    image_documents.append(image_doc)
                    
                except Exception as e:
                    print(f"  ⚠️  Error extracting image {img_index} from page {page_num}: {e}")
                    continue
    
    # Detect organisms and research types (from preprocess_docs.py)
    combined_text = "\n\n".join(full_text)
    title_lower = title.lower()
    text_sample_lower = combined_text[:2000].lower()
    
    # Check known organisms
    organisms = [
        org.capitalize() for org in ORGANISMS_SET
        if org in title_lower or org in text_sample_lower
    ]
    
    # Fallback: detect genus-species patterns
    if not organisms:
        matches = ORGANISM_PATTERN.findall(title + " " + combined_text[:500])
        if matches:
            organisms = [f"{genus} {species}" for genus, species in matches[:3]]
    
    # Detect research types
    research_types = []
    title_words = set(title_lower.split())
    
    if GENOMIC_KEYWORDS & title_words or any(kw in title_lower for kw in GENOMIC_KEYWORDS):
        research_types.append('genomic')
    if PHENOTYPIC_KEYWORDS & title_words:
        research_types.append('phenotypic')
    if META_ANALYSIS_KEYWORDS & title_words or 'meta-analysis' in title_lower:
        research_types.append('meta-analysis')
    if MICROBIOLOGY_KEYWORDS & title_words or any(kw in title_lower for kw in MICROBIOLOGY_KEYWORDS):
        research_types.append('microbiology')
    if SPACEFLIGHT_KEYWORDS & title_words or any(kw in title_lower for kw in SPACEFLIGHT_KEYWORDS):
        research_types.append('spaceflight')
    if TISSUE_KEYWORDS & title_words or any(kw in title_lower for kw in TISSUE_KEYWORDS):
        research_types.append('tissue-biology')
    
    # Create base metadata
    base_metadata = {
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
    
    # Now create text chunks with full metadata
    temp_doc = Document(
        page_content=combined_text,
        metadata=base_metadata
    )
    text_chunks = text_splitter.split_documents([temp_doc])
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(text_chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['type'] = 'text'
        
        # Estimate section position
        total_chunks = len(text_chunks)
        if total_chunks > 0:
            intro_threshold = int(total_chunks * 0.2)
            conclusion_threshold = int(total_chunks * 0.8)
            
            if i < intro_threshold:
                chunk.metadata['section_position'] = 'introduction'
            elif i > conclusion_threshold:
                chunk.metadata['section_position'] = 'conclusion'
            else:
                chunk.metadata['section_position'] = 'body'
    
    text_documents.extend(text_chunks)
    
    # Add metadata to image documents
    for img_doc in image_documents:
        img_doc.metadata.update({
            'organisms': base_metadata['organisms'],
            'research_types': base_metadata['research_types'],
            'pdf_author': base_metadata['pdf_author'],
            'pdf_subject': base_metadata['pdf_subject'],
        })
    
    doc.close()
    
    return {
        "text_documents": text_documents,
        "image_documents": image_documents,
        "total_text_chunks": len(text_documents),
        "total_images": len(image_documents),
        "pmc_id": pmc_id,
        "title": title,
        "organisms": base_metadata['organisms'],
        "research_types": base_metadata['research_types']
    }


def load_all_multimodal_documents(extract_images: bool = True) -> Dict[str, List[Document]]:
    """
    Load and process all PDFs with both text and images.
    
    Args:
        extract_images: Whether to extract images (default: True)
    
    Returns:
        Dictionary with separate lists for text and image documents
    """
    all_text_documents = []
    all_image_documents = []
    
    pdf_files = sorted(list(DOCUMENTS_PATH.glob("*.pdf")))
    
    print(f"\n📄 Found {len(pdf_files)} PDF files")
    print(f"🖼️  Image extraction: {'Enabled' if extract_images else 'Disabled'}\n")
    print("-" * 80)
    
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"[{i:2d}/{len(pdf_files)}] Processing: {pdf_path.name[:65]}...")
            
            result = extract_multimodal_from_pdf(pdf_path, extract_images=extract_images)
            
            all_text_documents.extend(result["text_documents"])
            all_image_documents.extend(result["image_documents"])
            
            # Show extracted metadata
            organisms = result.get('organisms', ['unknown'])
            types = result.get('research_types', ['general'])
            
            print(f"        └─ Text chunks: {result['total_text_chunks']}")
            if extract_images:
                print(f"        └─ Images: {result['total_images']}")
            print(f"        └─ Organisms: {', '.join(organisms[:3])}")
            print(f"        └─ Types: {', '.join(types)}")
            
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
    
    print("-" * 80)
    print(f"\n✅ Processing complete!")
    print(f"   📝 Total text chunks: {len(all_text_documents)}")
    if extract_images:
        print(f"   🖼️  Total images: {len(all_image_documents)}")
    print(f"   📊 Total documents: {len(all_text_documents) + len(all_image_documents)}\n")
    
    return {
        "text_documents": all_text_documents,
        "image_documents": all_image_documents,
        "all_documents": all_text_documents + all_image_documents
    }


def merge_documents_for_embedding(text_docs: List[Document], image_docs: List[Document]) -> List[Document]:
    """
    Merge text and image documents, preserving order and metadata.
    
    Returns a single list ready for CLIP embedding.
    """
    all_docs = []
    
    # Add text documents
    for doc in text_docs:
        doc.metadata["modality"] = "text"
        all_docs.append(doc)
    
    # Add image documents
    for doc in image_docs:
        doc.metadata["modality"] = "image"
        all_docs.append(doc)
    
    return all_docs
