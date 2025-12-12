#!/usr/bin/env python3
"""
Data Preparation Script - Creates chunks AND combined file
Processes ArXiv JSON → CSV chunks → Combined CSV
"""

import json
import pandas as pd
import argparse
from pathlib import Path


def prepare_arxiv_data(json_path, max_papers=None, chunk_size=80000):
    """
    Process ArXiv JSON and create both chunks and combined file.
    
    """
    print("="*70)
    print("📚 ArXiv Data Preparation Pipeline")
    print("="*70)
    
    # Setup directories
    Path("data/csv_files").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Reading: {json_path}")
    
    all_data = []
    chunk_num = 0
    current_chunk = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_papers and i >= max_papers:
                break
            
            try:
                paper = json.loads(line)
                
                # Prepare text
                title = paper.get('title', '').strip().replace('\n', ' ')
                abstract = paper.get('abstract', '').strip().replace('\n', ' ')
                prepared_text = f"{title} {{title}} {abstract}"
                
                record = {
                    'id': paper.get('id', f'unknown_{i}'),
                    'prepared_text': prepared_text
                }
                
                current_chunk.append(record)
                all_data.append(record)
                
                # Save chunk when full
                if len(current_chunk) >= chunk_size:
                    start_idx = chunk_num * chunk_size
                    end_idx = start_idx + len(current_chunk)
                    chunk_path = f"data/csv_files/metadata_{start_idx}_{end_idx}.csv"
                    
                    pd.DataFrame(current_chunk).to_csv(chunk_path, index=False)
                    print(f"✅ Saved chunk {chunk_num}: {chunk_path} ({len(current_chunk)} papers)")
                    
                    current_chunk = []
                    chunk_num += 1
                
                if (i + 1) % 10000 == 0:
                    print(f"   Processed {i + 1} papers...")
                    
            except json.JSONDecodeError:
                continue
    
    # Save last chunk if not empty
    if current_chunk:
        start_idx = chunk_num * chunk_size
        end_idx = start_idx + len(current_chunk)
        chunk_path = f"data/csv_files/metadata_{start_idx}_{end_idx}.csv"
        
        pd.DataFrame(current_chunk).to_csv(chunk_path, index=False)
        print(f"✅ Saved chunk {chunk_num}: {chunk_path} ({len(current_chunk)} papers)")
    
    # Save combined file
    print(f"\n📊 Creating combined file...")
    combined_df = pd.DataFrame(all_data)
    combined_path = "data/processed/metadata_all.csv"
    combined_df.to_csv(combined_path, index=False)
    
    print(f"✅ Saved combined: {combined_path} ({len(all_data)} total papers)")
    
    print("\n" + "="*70)
    print("✅ DATA PREPARATION COMPLETE")
    print("="*70)
    print(f"Chunks: {chunk_num + 1} files in data/csv_files/")
    print(f"Combined: {combined_path}")
    print(f"\nNext: Run generate_embeddings.py on each chunk")
    print("="*70)
    
    return combined_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", default="data/raw/arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--max-papers", type=int, help="Max papers to process (for testing)")
    parser.add_argument("--chunk-size", type=int, default=80000)
    
    args = parser.parse_args()
    prepare_arxiv_data(args.json_path, args.max_papers, args.chunk_size)
