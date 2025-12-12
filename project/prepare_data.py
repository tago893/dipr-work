#!/usr/bin/env python3
"""Simple data preparation script."""

import json
import pandas as pd
from pathlib import Path
import argparse


def process_arxiv_json(input_file, output_dir, max_records=None):
    """Process ArXiv JSON file - simple version like the notebook."""
    print(f"Processing {input_file}...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    data = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
                
            try:
                paper = json.loads(line)
                data.append({
                    'id': paper.get('id', ''),
                    'title': paper.get('title', ''),
                    'abstract': paper.get('abstract', ''),
                    'authors': str(paper.get('authors', '')),
                    'categories': paper.get('categories', '')
                })
                
                if i % 10000 == 0:
                    print(f"Processed {i} records...")
                    
            except:
                continue
    
    # Create DataFrame and prepared_text
    df = pd.DataFrame(data)
    df['prepared_text'] = df['title'] + ' {title} ' + df['abstract']
    
    # Save
    output_file = Path(output_dir) / "arxiv_processed.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} records to {output_file}")


# def create_sample_data(output_dir, n_samples=1000):
#     """Create sample data for testing."""
#     print(f"Creating {n_samples} sample records...")
    
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     data = []
#     for i in range(n_samples):
#         data.append({
#             'id': f"2024.{i:05d}",
#             'title': f"Sample Paper {i+1}: Research Topic",
#             'abstract': f"This is sample abstract {i+1} discussing important research concepts.",
#             'authors': f"Author {i+1}",
#             'categories': 'cs.AI',
#             'prepared_text': f"Sample Paper {i+1}: Research Topic {{title}} This is sample abstract {i+1}."
#         })
    
#     df = pd.DataFrame(data)
#     output_file = Path(output_dir) / "arxiv_sample.csv"
#     df.to_csv(output_file, index=False)
#     print(f"Created sample data: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Input JSON file")
    parser.add_argument("--output", "-o", default="data/processed", help="Output directory")
    parser.add_argument("--max-records", "-m", type=int, help="Max records to process")
    parser.add_argument("--sample", "-s", action="store_true", help="Create sample data")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_data(args.output, args.sample_size)
    elif args.input:
        process_arxiv_json(args.input, args.output, args.max_records)
    else:
        print("Use --sample for sample data or --input <file> for real data")
        parser.print_help()