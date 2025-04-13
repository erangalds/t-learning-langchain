import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_folder = os.path.join(project_root, "sample-data")
text_file_name = "test.txt" # Ensure this file exists in sample-data
text_file_path = os.path.join(data_folder, text_file_name)

print(f"Attempting to load text file from: {text_file_path}")

# Check if the file exists
if not os.path.isfile(text_file_path):
    print(f"Error: Text file not found at {text_file_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()
# --- End Configuration ---

try:
    # 1. Load the document using TextLoader
    # TextLoader loads the entire file into a single Document object by default
    loader = TextLoader(text_file_path, encoding='utf-8')
    documents = loader.load() # Returns a list containing one Document

    print(f"\nLoaded {len(documents)} document(s) using TextLoader.")
    if not documents:
        print("No document loaded, exiting.")
        exit()

    print(f"Characters in loaded document: {len(documents[0].page_content)}")

    # 2. Initialize the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Example chunk size
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # 3. Split the loaded document(s)
    # split_documents expects a list of Documents and returns a list of Documents (chunks)
    chunks = text_splitter.split_documents(documents)

    print(f"\nSplit the document into {len(chunks)} chunks.")

    if chunks:
        print("\n--- First 3 Chunks ---")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[Chunk {i+1}] (Length: {len(chunk.page_content)} characters)")
            # Note: Access content via chunk.page_content for Document objects
            print(f"Content:\n'{chunk.page_content}'")
            print(f"Metadata: {chunk.metadata}") # Metadata usually includes source
            print("-" * 20)

except FileNotFoundError:
    print(f"Error: File not found at {text_file_path}")
except Exception as e:
    print(f"\nAn error occurred: {e}")

