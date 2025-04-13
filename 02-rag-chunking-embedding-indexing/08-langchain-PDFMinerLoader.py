# Script using PDFMinerLoader
# Requires: pip install pdfminer.six

import os
from langchain_community.document_loaders import PDFMinerLoader

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
pdf_folder = os.path.join(project_root, "sample-data")
pdf_file_name = "test.pdf" # Ensure this file exists in sample-data
pdf_file_path = os.path.join(pdf_folder, pdf_file_name)

print(f"Attempting to load PDF using PDFMinerLoader from: {pdf_file_path}")

# Check if the file exists
if not os.path.isfile(pdf_file_path):
    print(f"Error: PDF file not found at {pdf_file_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()
# --- End Configuration ---

# Initialize the PDFMinerLoader
# Note: PDFMinerLoader loads the *entire* PDF as a single Document by default
loader = PDFMinerLoader(pdf_file_path)

try:
    # Load the PDF content into documents (usually just one document)
    documents = loader.load()

    print(f"\nSuccessfully loaded {len(documents)} document(s) using PDFMinerLoader.")
    print("(Note: PDFMinerLoader often loads the entire PDF as one document)")

    if documents:
        # Print info about the first (and likely only) document
        print(f"\n--- Content Snippet (Document 1) ---")
        print(f"{documents[0].page_content[:500]}...") # Print first 500 characters
        print("---")

        # Print metadata of the first document
        print(f"\nMetadata (Document 1): {documents[0].metadata}")
        # Metadata typically includes 'source'

except Exception as e:
    print(f"\nAn error occurred during PDF loading with PDFMinerLoader: {e}")

