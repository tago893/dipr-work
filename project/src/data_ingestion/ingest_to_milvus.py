"""Ingest embeddings and metadata into Milvus database."""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.vector_db.milvus_service import MilvusClient


def ingest_to_milvus(
    embeddings_path: str,
    metadata_path: str,
    milvus_uri: str = "data/milvus/arxiv_papers.db",
    collection_name: str = "arxiv_papers",
    batch_size: int = 1000,
    drop_existing: bool = False
):
    """
    Ingest embeddings and metadata into Milvus.

    """
    print("MILVUS DATA INGESTION")
    
    # Load embeddings
    import glob
    
    # Load ALL embeddings from directory or glob pattern
    if '*' in embeddings_path:
        search_pattern = embeddings_path
    elif os.path.isdir(embeddings_path):
        search_pattern = os.path.join(embeddings_path, "embeddings_*.npy")
    else:
        # Fallback for single file, but we try to be smart if it looks like a file in a dir of others
        if "embeddings_" in embeddings_path and os.path.exists(embeddings_path):
             # User passed one file, but maybe wants them all? 
             # For now, let's trust the path unless it's a directory
             search_pattern = embeddings_path
        else:
             search_pattern = embeddings_path

    # If it's a specific single file, just load it. 
    # BUT user said "I want all the embedding", so I will assume directory logic if possible.
    if os.path.isfile(embeddings_path) and not '*' in embeddings_path:
         # Check if we should actually look for siblings? 
         # The user said "remove single files I want all...". 
         # So if input is a single file, I'll switch to directory search.
         parent_dir = os.path.dirname(embeddings_path) or '.'
         search_pattern = os.path.join(parent_dir, "embeddings_*.npy")
    
    embedding_files = sorted(glob.glob(search_pattern))
    if not embedding_files:
        print(f"No files found matching: {search_pattern}")
        return False
        
    print(f"Found {len(embedding_files)} embedding files. Loading all...")
    embedding_list = []
    for f in embedding_files:
        print(f"  Loading {os.path.basename(f)}...")
        embedding_list.append(np.load(f))
    
    embeddings = np.vstack(embedding_list)
    
    # Load metadata
    print("Loading metadata...")
    metadata_df = pd.read_csv(metadata_path, dtype={'id': str})
    
    # Verify alignment - if metadata is larger, we slice it
    if len(embeddings) != len(metadata_df):
        print(f"\n ⚠️  Count mismatch: Embeddings={len(embeddings)}, Metadata={len(metadata_df)}")
        if len(metadata_df) > len(embeddings):
            print(f"    Slicing metadata to match first {len(embeddings)} entries...")
            metadata_df = metadata_df.iloc[:len(embeddings)]
        else:
            print("    ERROR: Metadata is smaller than embeddings!")
            return False
        
    # Convert metadata to list of dicts - Only 'id' is used by MilvusClient.insert_embeddings_batch
    # Schema check confirms we only store: id (auto), arxiv_id, embedding_index, vector
    arxiv_data = []
    # We can iterate faster by just pulling the column
    ids = metadata_df['id'].astype(str).tolist()
    
    # Pre-allocate dictionary list for speed
    arxiv_data = [{'id': arxiv_id} for arxiv_id in ids]
    
    # Connect to Milvus
    print(f"\n  Connecting to Milvus: {milvus_uri}")
    client = MilvusClient(uri=milvus_uri)
    print(" Connected")
    
    # Check existing collection
    if client.client.has_collection(collection_name):
        client.load_collection(collection_name)
        stats = client.get_stats()
        existing_count = stats.get('num_entities', 0)
        
        print(f"\  Collection '{collection_name}' already exists")
        print(f"   Documents: {existing_count}")
        
        if existing_count > 0 and not drop_existing:
            # Collection has data and user didn't request drop
            print("\n DATA ALREADY PRESENT - Using existing data!")
            print("   (Use --drop-existing to rebuild)")
            return True  # Skip insertion
        elif drop_existing:
            # User wants to rebuild
            print("Dropping and recreating...")
            client.create_collection(collection_name, embeddings.shape[1], drop_existing=True)
            print("Collection recreated")
        # else: collection exists but is empty, continue to insert
    else:
        # New collection
        print(f"\n  Creating collection: {collection_name}")
        client.create_collection(collection_name, embeddings.shape[1])
        print(" Created")
    
    # Create index
    print("\nCreating index...")
    client.create_index(collection_name)
    print(" Index created")
    
    # Insert data
    print(f"\n Inserting {len(embeddings)} documents...")
    print(f"   Batch size: {batch_size}")
    
    success = client.insert_embeddings_batch(
        embeddings=embeddings,
        arxiv_data=arxiv_data,
        start_index=0,
        batch_size=batch_size
    )
    
    if not success:
        print("Insertion failed, try again!")
        return False
    
    print(" Data inserted successfully!")
    
    # Verify
    stats = client.get_stats()
    final_count = stats.get('num_entities', 0)
    print(f"   Total documents: {final_count}")
    
    if final_count >= len(embeddings):
        return True
    else:
        print(f"\n Partial insertion: {final_count}/{len(embeddings)}")
        return False

