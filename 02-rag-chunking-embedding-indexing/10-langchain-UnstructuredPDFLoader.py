# Script using UnstructuredPDFLoader
# Requires: pip install "unstructured[pdf]"
# May require system dependencies like poppler-utils (Linux) or poppler (macOS/Windows)
# For OCR on images within PDFs, may also need tesseract

import os
from langchain_community.document_loaders import UnstructuredPDFLoader

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
pdf_folder = os.path.join(project_root, "sample-data")
pdf_file_name = "test.pdf" # Ensure this file exists in sample-data
pdf_file_path = os.path.join(pdf_folder, pdf_file_name)

print(f"Attempting to load PDF using UnstructuredPDFLoader from: {pdf_file_path}")

# Check if the file exists
if not os.path.isfile(pdf_file_path):
    print(f"Error: PDF file not found at {pdf_file_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()
# --- End Configuration ---

# Initialize the UnstructuredPDFLoader
# - mode: "single" concatenates all elements into one Document.
#         "elements" creates a Document per element (paragraph, table, etc.).
#         "paged" aims to create a Document per page (might require specific dependencies).
# "elements" mode is often useful for understanding document structure.
loader = UnstructuredPDFLoader(
    pdf_file_path,
    mode="elements", # Try "single" or "paged" as alternatives
    strategy="fast", # Use "hi_res" for potentially better but slower processing (may need more deps)
    )

try:
    # Load the PDF content into documents based on the mode
    documents = loader.load()

    print(f"\nSuccessfully loaded {len(documents)} elements/documents using UnstructuredPDFLoader (mode='{loader.mode}').")

    if documents:
        # Print info about the first element/document
        print(f"\n--- Content Snippet (First Element/Document) ---")
        print(f"{documents[0].page_content[:500]}...") # Print first 500 characters
        print("---")

        # Print metadata of the first element/document
        # Metadata can be rich, including element type, page number, etc.
        print(f"\nMetadata (First Element/Document): {documents[0].metadata}")

except Exception as e:
    print(f"\nAn error occurred during PDF loading with UnstructuredPDFLoader: {e}")
    print("Ensure all required dependencies for 'unstructured[pdf]' and potentially system libraries (poppler, tesseract) are installed.")

# Key points about UnstructuredPDFLoader:

# Dependencies: It has more complex dependencies than the others. Make sure you run pip install "unstructured[pdf]". You might encounter further dependency issues related to system libraries (poppler, tesseract) depending on your setup and the PDF content.
# mode Parameter: This significantly changes the output.
# "elements" (often the default or most common): Tries to break the PDF down into logical chunks (paragraphs, titles, list items, potentially tables) and creates a separate LangChain Document for each. This is great for understanding structure but results in many small documents.
# "single": Combines all extracted text into one single Document. Simpler, but loses the element-level granularity.
# "paged": Attempts to group elements by page.
# strategy Parameter: Controls the processing method ("fast", "ocr_only", "hi_res"). "hi_res" often yields better results for complex layouts but is slower and might require more setup (like detectron2).
# Metadata: The metadata associated with each Document when using "elements" mode can be very informative, often including the type of element detected (e.g., Header, NarrativeText, ListItem).
# This loader is powerful when dealing with PDFs that aren't just simple text flows, especially if you need to treat tables or specific sections differently later in your RAG pipeline.