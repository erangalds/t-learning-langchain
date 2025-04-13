from langchain_community.document_loaders import TextLoader

loader = TextLoader("sample-data/test.txt",encoding="utf-8")

# Check if the file exists
import os
if not os.path.isfile(loader.file_path):
    print(f"Error: File not found at {loader.file_path}")
    # Handle the error appropriately, maybe exit or raise an exception
    exit()
try:
    # Load the documents
    # The loader will read the file and return a list of Document objects
    documents = loader.load()
    
    if documents:
        # Print the number of documents loaded
        print(f"\nLoaded {len(documents)} documents.")
        # Print the first document content (or a snippet)
        print(f"\nFirst document content snippet:\n---\n{documents[0].page_content[:200]}...\n---")
        # Print Document metadata for the first document
        print(f"\nFirst document metadata: {documents[0].metadata}")
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit


