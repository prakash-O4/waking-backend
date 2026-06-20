#!/usr/bin/env python3
"""
Script to delete documents from Pinecone by source_id metadata.
This allows you to remove specific ingested documents.
"""
import sys
import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def delete_by_source_id(source_id, index_name='wakil-g'):
    """
    Delete all documents with a specific source_id from Pinecone.

    Args:
        source_id: The source_id metadata value to filter by
        index_name: Name of the Pinecone index (default: 'wakil-g')
    """
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(index_name)

        print(f"Connected to Pinecone index: {index_name}")
        print(f"Attempting to delete documents with source_id: {source_id}")
        print("-" * 50)

        # Delete vectors by metadata filter
        # Note: In newer Pinecone, delete by metadata filter
        delete_response = index.delete(
            filter={
                "source_id": {"$eq": source_id}
            }
        )

        print("Delete operation completed!")
        print(f"Response: {delete_response}")
        print("-" * 50)
        print(f"All documents with source_id '{source_id}' have been deleted.")

    except Exception as e:
        print(f"Error during deletion: {e}")
        print("\nNote: Make sure the PINECONE_API_KEY is set in your .env file")
        raise

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Delete documents from Pinecone by source_id'
    )
    parser.add_argument(
        'source_id',
        type=str,
        help='The source_id to delete (e.g., case_law_2024)'
    )
    parser.add_argument(
        '--index',
        type=str,
        default='wakil-g',
        help='Pinecone index name (default: wakil-g)'
    )

    args = parser.parse_args()

    # Confirm before deletion
    print(f"\nWARNING: This will delete all documents with source_id: {args.source_id}")
    confirm = input("Are you sure you want to proceed? (yes/no): ")

    if confirm.lower() == 'yes':
        delete_by_source_id(args.source_id, args.index)
    else:
        print("Deletion cancelled.")

if __name__ == "__main__":
    main()
