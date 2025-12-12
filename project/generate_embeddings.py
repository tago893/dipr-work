#!/usr/bin/env python3
"""
Step 2: Generate Embeddings
Load prepared CSV and generate embeddings using allenai-specter.

Usage:
    python generate_embeddings.py --csv data/csv_files/metadata_0_80000.csv
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import torch

from embedding.embedding_service import SimpleEmbedder


def generate_embeddings_from_csv(csv_path, model_name="allenai-specter", 
                                 batch_size=1000, output_dir="data/embeddings",
                                 save_every=25):
    """
    Generate embeddings from CSV file with partial saves.
    
    """
    print(f"\n📂 Loading CSV: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    if 'prepared_text' not in df.columns:
        print(f"❌ Error: CSV must have 'prepared_text' column")
        print(f"   Found columns: {df.columns.tolist()}")
        return None
    
    texts = df['prepared_text'].tolist()
    print(f"✅ Loaded {len(texts)} texts")
    
    # Extract batch info from CSV filename (e.g., metadata_0_80000.csv)
    csv_name = Path(csv_path).stem  # metadata_0_80000
    batch_info = csv_name.replace('metadata_', '')  # 0_80000
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Using GPU for embeddings")
    else:
        print(f"\n⚠️  No GPU detected - using CPU (will be slower)")
    
    # Generate embeddings with partial saves (same as your notebook)
    print(f"\n🧠 Generating embeddings with {model_name}")
    print(f"   Batch size: {batch_size}")
    print(f"   Partial saves every {save_every} batches")
    
    os.makedirs(output_dir, exist_ok=True)
    
    embedder = SimpleEmbedder(model_name=model_name)
    embeddings = []
    
    # Process in batches with partial saves
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_embeddings = embedder.model.encode(batch_texts, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
            
            batch_num = i // batch_size + 1
            total_batches = (len(texts) - 1) // batch_size + 1
            print(f"Processed batch {batch_num}/{total_batches}")
            
            # Save partial progress every N batches (like your notebook)
            if batch_num % save_every == 0:
                partial_path = f"{output_dir}/embeddings_partial_{batch_info}.npy"
                np.save(partial_path, np.array(embeddings))
                print(f"💾 Saved partial progress at batch {batch_num}: {partial_path}")
                
        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            continue
    
    embeddings = np.array(embeddings)
    print(f"✅ Generated embeddings shape: {embeddings.shape}")
    
    # Save final embeddings
    embeddings_path = f"{output_dir}/embeddings_{batch_info}.npy"
    np.save(embeddings_path, embeddings)
    
    print(f"✅ Saved final embeddings: {embeddings_path}")
    
    # Clean up partial file if exists
    partial_path = f"{output_dir}/embeddings_partial_{batch_info}.npy"
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print(f"🗑️  Removed partial file: {partial_path}")
    
    return embeddings_path


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from prepared CSV")
    
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file (e.g., data/csv_files/metadata_0_80000.csv)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="allenai-specter",
        help="Embedding model name"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Output directory for embeddings"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"🧠 Embedding Generation")
    print("="*70)
    print(f"CSV: {args.csv}")
    print(f"Model: {args.model}")
    print("="*70)
    
    embeddings_path = generate_embeddings_from_csv(
        csv_path=args.csv,
        model_name=args.model,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    if embeddings_path:
        print("\n" + "="*70)
        print("✅ EMBEDDING GENERATION COMPLETE")
        print("="*70)
        print(f"Embeddings saved: {embeddings_path}")
        print("="*70)


if __name__ == "__main__":
    main()
