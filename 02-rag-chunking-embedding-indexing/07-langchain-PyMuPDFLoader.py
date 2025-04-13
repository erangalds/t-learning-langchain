# Script using PyMuPDFLoader
# Requires: pip install pymupdf

import os
from langchain_community.document_loaders import PyMuPDFLoader

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
pdf_folder = os.path.join(project_root, "sample-data")
pdf_file_name = "test.pdf" # Ensure this file exists in sample-data
pdf_file_path = os.path.join(pdf_folder, pdf_file_name)

print(f"Attempting to load PDF using PyMuPDFLoader from: {pdf_file_path}")

# Check if the file exists
if not os.path.isfile(pdf_file_path):
    print(f"Error: PDF file not found at {pdf_file_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()
# --- End Configuration ---

# Initialize the PyMuPDFLoader
# Loads each page as a separate Document
loader = PyMuPDFLoader(pdf_file_path)

try:
    # Load the PDF pages into documents
    pages = loader.load()

    print(f"\nSuccessfully loaded {len(pages)} pages using PyMuPDFLoader.")

    if pages:
        # Print info about the first page (first document)
        print(f"\n--- Content Snippet (Page 1) ---")
        print(f"{pages[0].page_content[:500]}...") # Print first 500 characters
        print("---")

        # Print metadata of the first page
        print(f"\nMetadata (Page 1): {pages[0].metadata}")
        # Metadata includes 'source', 'file_path', 'page', 'total_pages', etc.

except Exception as e:
    print(f"\nAn error occurred during PDF loading with PyMuPDFLoader: {e}")

