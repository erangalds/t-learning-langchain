# Script using PyPDFium2Loader
# Requires: pip install pypdfium2

import os
from langchain_community.document_loaders import PyPDFium2Loader

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
pdf_folder = os.path.join(project_root, "sample-data")
pdf_file_name = "test.pdf" # Ensure this file exists in sample-data
pdf_file_path = os.path.join(pdf_folder, pdf_file_name)

print(f"Attempting to load PDF using PyPDFium2Loader from: {pdf_file_path}")

# Check if the file exists
if not os.path.isfile(pdf_file_path):
    print(f"Error: PDF file not found at {pdf_file_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()
# --- End Configuration ---

# Initialize the PyPDFium2Loader
# Loads each page as a separate Document
loader = PyPDFium2Loader(pdf_file_path)

try:
    # Load the PDF pages into documents
    pages = loader.load()

    print(f"\nSuccessfully loaded {len(pages)} pages using PyPDFium2Loader.")

    if pages:
        # Print info about the first page (first document)
        print(f"\n--- Content Snippet (Page 1) ---")
        print(f"{pages[0].page_content[:500]}...") # Print first 500 characters
        print("---")

        # Print metadata of the first page
        print(f"\nMetadata (Page 1): {pages[0].metadata}")
        # Metadata typically includes 'source' and 'page'

except Exception as e:
    print(f"\nAn error occurred during PDF loading with PyPDFium2Loader: {e}")

