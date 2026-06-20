#!/usr/bin/env python3
"""
Script to ingest Case Law PDF into Pinecone with unique metadata.
This allows for later deletion/filtering of this specific document.
"""
import sys
import os

# Add parent directory to path to import from app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.utils.ingest_data import init_pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Define the path to case law PDF
    case_law_path = 'source/Case Law.pdf'

    # Define a unique source ID for this ingestion
    # This will be added as metadata to all chunks
    source_id = 'case_law_2024'

    print(f"Starting ingestion of: {case_law_path}")
    print(f"Source ID: {source_id}")
    print("-" * 50)

    # Ingest the document with unique metadata
    init_pinecone(case_law_path, source_id=source_id)

    print("-" * 50)
    print("Ingestion complete!")
    print(f"To delete this document later, use source_id: {source_id}")

if __name__ == "__main__":
    main()
