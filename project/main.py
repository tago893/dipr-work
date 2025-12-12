"""Main entry point for ArXiv RAG System."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vector_db.milvus_service import MilvusClient
from embedding.embedding_service import SimpleEmbedder
from generation.llm_service import LLMService
from retrieval.retriever import SemanticRetriever
from generation.milvus_rag_pipeline import MilvusRAGPipeline


def verify_data_files():
    """Check if required data files exist."""
    required_files = {
        "CSV": "data/processed/metadata_all.csv",
        "Embeddings": "data/embeddings/embeddings_0_80000.npy",
    }
    
    missing = [name for name, path in required_files.items() if not Path(path).exists()]
    
    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        print("\nRun these commands first:")
        print("  1. python prepare_arxiv_data.py --json-path data/raw/arxiv-metadata-oai-snapshot.json")
        print("  2. python generate_embeddings.py --csv data/csv_files/metadata_0_80000.csv")
        return False
    
    return True


def ensure_milvus_ready():
    """Ensure Milvus database is ready with data."""
    from vector_db.milvus_service import MilvusClient
    from data_ingestion.ingest_to_milvus import ingest_to_milvus
    
    milvus_uri = "data/milvus_data/milvus_demo.db"
    collection_name = "milvus_demo"
    
    # Check if database exists and has data
    if Path(milvus_uri).exists():
        try:
            client = MilvusClient(uri=milvus_uri)
            if client.load_collection(collection_name):
                stats = client.get_stats()
                if stats.get('num_entities', 0) > 0:
                    print(f"✅ Milvus ready ({stats['num_entities']} documents)")
                    return True
        except:
            pass
    
    # Need to ingest
    print("⚙️  Ingesting data into Milvus...")
    try:
        success = ingest_to_milvus(
            embeddings_path="data/embeddings",
            metadata_path="data/processed/metadata_all.csv",
            milvus_uri=milvus_uri,
            collection_name=collection_name
        )
        return success
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return False


def run_application(provider="gemini"):
    """Initialize and run the RAG application."""
   
    
    print(f"\n🚀 Starting RAG System (Provider: {provider.upper()})")
    
    # Initialize components
    milvus_client = MilvusClient(uri="data/milvus_data/milvus_demo.db")
    milvus_client.load_collection("milvus_demo")
    
    embedder = SimpleEmbedder(model_name="allenai-specter")
    
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY not found in .env")
    llm_service = LLMService(provider=provider, api_key=api_key)
    
    retriever = SemanticRetriever(milvus_client)
    rag_pipeline = MilvusRAGPipeline(retriever, embedder, llm_service)
    
    # Interactive mode
    print("✅ System ready\n" + "="*60)
    print("💬 Ask questions (type 'exit' to quit)")
    print("="*60)
    
    while True:
        try:
            question = input("\n🔍 Query: ").strip()
            
            if not question or question.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
            
            result = rag_pipeline.query(question, top_k=5)
            
            print(f"\n💡 Answer:\n{'-'*60}")
            print(result.generated_response.content)
            print('-'*60)
            
            print(f"\n📚 Retrieved Papers:")
            for i, doc in enumerate(result.retrieved_documents, 1):
                print(f"   {i}. {doc['arxiv_id']} (score: {doc['similarity_score']:.4f})")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ArXiv RAG System")
    parser.add_argument("--provider", default="gemini", 
                       choices=["gemini", "openai", "huggingface"])
    parser.add_argument("--skip-check", action="store_true", 
                       help="Skip data verification")
    
    args = parser.parse_args()
    load_dotenv()
    
    # Verify data
    if not args.skip_check:
        if not verify_data_files() or not ensure_milvus_ready():
            sys.exit(1)
    
    # Run
    run_application(provider=args.provider)


if __name__ == "__main__":
    main()
