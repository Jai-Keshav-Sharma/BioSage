"""
Build Qdrant Vector Store with CLIP Embeddings (Text + Images)
This script loads, processes, and indexes NASA space biology papers
including both text content and images from PDFs

OPTIMIZED: Processes PDFs in batches to avoid memory issues
"""

from pathlib import Path
from src.preprocess_multimodal import extract_multimodal_from_pdf
from src.create_vectorstore import create_qdrant_store_batched, initialize_qdrant_collection
from src.config import BATCH_SIZE, DOCUMENTS_PATH
import gc


def build_vectorstore(extract_images: bool = True, pdf_batch_size: int = 50):
    """
    Main pipeline to build the multimodal vector store with batched processing.
    
    Args:
        extract_images: Whether to extract and embed images (default: True)
        pdf_batch_size: Number of PDFs to process at once (default: 50)
    """
    
    print("=" * 80)
    print("🚀 NASA Space Biology Papers - Multimodal Vector Store Builder")
    print("   OPTIMIZED: Batched processing to prevent memory issues")
    print("=" * 80)
    
    # Step 1: Initialize empty Qdrant collection
    print("\n� STEP 1: Initializing Qdrant collection...")
    collection_created = initialize_qdrant_collection()
    
    if not collection_created:
        print("❌ Failed to create collection. Check Docker Qdrant is running.")
        return
    
    # Step 2: Get all PDF files
    pdf_files = sorted(list(DOCUMENTS_PATH.glob("*.pdf")))
    
    if not pdf_files:
        print("❌ No PDF files found! Check your documents folder.")
        return
    
    print(f"\n📄 Found {len(pdf_files)} PDF files")
    print(f"🖼️  Image extraction: {'Enabled' if extract_images else 'Disabled'}")
    print(f"📦 Processing in batches of {pdf_batch_size} PDFs")
    print()
    
    # Step 3: Process PDFs in batches
    total_pdfs = len(pdf_files)
    num_batches = (total_pdfs + pdf_batch_size - 1) // pdf_batch_size
    
    total_text_chunks = 0
    total_images = 0
    
    for batch_num in range(num_batches):
        start_idx = batch_num * pdf_batch_size
        end_idx = min(start_idx + pdf_batch_size, total_pdfs)
        batch_files = pdf_files[start_idx:end_idx]
        
        print("=" * 80)
        print(f"� BATCH {batch_num + 1}/{num_batches}: Processing PDFs {start_idx + 1}-{end_idx}")
        print("=" * 80)
        
        # Process this batch of PDFs
        batch_documents = []
        
        for i, pdf_path in enumerate(batch_files, start=start_idx + 1):
            try:
                print(f"[{i:3d}/{total_pdfs}] Processing: {pdf_path.name[:65]}...")
                
                result = extract_multimodal_from_pdf(pdf_path, extract_images=extract_images)
                
                # Collect all documents from this PDF
                batch_documents.extend(result["text_documents"])
                batch_documents.extend(result["image_documents"])
                
                total_text_chunks += result['total_text_chunks']
                total_images += result['total_images']
                
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
                continue
        
        # Add this batch to Qdrant
        if batch_documents:
            print(f"\n🔮 Adding {len(batch_documents)} documents to Qdrant...")
            success = create_qdrant_store_batched(batch_documents, batch_size=BATCH_SIZE)
            
            if success:
                print(f"✅ Batch {batch_num + 1}/{num_batches} added successfully!")
            else:
                print(f"⚠️  Some documents in batch {batch_num + 1} may have failed")
            
            # Clear memory
            del batch_documents
            gc.collect()
        
        print()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎉 Multimodal vector store built successfully!")
    print("=" * 80)
    print(f"\n📊 Final Statistics:")
    print(f"   � Total text chunks: {total_text_chunks:,}")
    if extract_images:
        print(f"   🖼️  Total images: {total_images:,}")
    print(f"   📊 Total documents: {total_text_chunks + total_images:,}")
    print("\n�💡 What you can do now:")
    print("   • Search with text queries → find relevant text AND images")
    print("   • CLIP embeddings enable unified semantic search")
    print("   • Images and text are in the same vector space")
    print("   • Use for multimodal RAG pipelines\n")
    
    return True


if __name__ == "__main__":
    # Set extract_images=False to only process text (faster)
    build_vectorstore(extract_images=True)
