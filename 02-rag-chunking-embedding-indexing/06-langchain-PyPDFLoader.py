import os
from langchain_community.document_loaders import PyPDFLoader

# --- Configuration ---
# Construct the path relative to the script's location
# Assumes this script is in a folder one level below the project root
# Adjust the relative path if your script is located elsewhere
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Go one level up
pdf_folder = os.path.join(project_root, "sample-data")
pdf_file_name = "test.pdf" # <--- CHANGE THIS to your actual PDF file name
pdf_file_path = os.path.join(pdf_folder, pdf_file_name)

print(f"Attempting to load PDF from: {pdf_file_path}")

# Check if the file exists
if not os.path.isfile(pdf_file_path):
    print(f"Error: PDF file not found at {pdf_file_path}")
    print("Please ensure the file exists and the path is correct.")
    # Exit or handle as needed
    exit()
# --- End Configuration ---

# Initialize the PyPDFLoader
# By default, it loads each page as a separate Document
loader = PyPDFLoader(pdf_file_path)

try:
    # Load the PDF pages into documents
    # Each document in the list corresponds to one page of the PDF
    pages = loader.load()

    print(f"\nSuccessfully loaded {len(pages)} pages from the PDF.")

    if pages:
        # Print info about the first page (first document)
        print(f"\n--- Content Snippet (Page 1) ---")
        # Access page_content attribute of the Document object
        print(f"{pages[0].page_content[:500]}...") # Print first 500 characters
        print("---")

        # Print metadata of the first page
        print(f"\nMetadata (Page 1): {pages[0].metadata}")
        # Metadata typically includes 'source' (file path) and 'page' (page number, 0-indexed)

    # You could also use load_and_split() to directly chunk the document
    # from langchain_text_splitters import CharacterTextSplitter
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # documents = loader.load_and_split(text_splitter=text_splitter)
    # print(f"\nLoaded and split into {len(documents)} chunks.")
    # if documents:
    #     print(f"\nFirst chunk content snippet:\n{documents[0].page_content[:200]}...")


except Exception as e:
    print(f"\nAn error occurred during PDF loading: {e}")

