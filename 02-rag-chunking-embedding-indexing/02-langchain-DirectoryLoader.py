import os 
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Define the path to the directory containing the text files
# Using os.path.join for better cross-platform compatibility
data_directory = os.path.join("sample-data", "super-heros") 

print(f"Attempting to load files from: {os.path.abspath(data_directory)}") # Added for debugging path issues

# Check if the directory exists
if not os.path.isdir(data_directory):
    print(f"Error: Directory not found at {data_directory}")
    # Handle the error appropriately, maybe exit or raise an exception
    exit()

# Configure the DirectoryLoader
# - path: The directory to load from
# - glob: A pattern to match files (e.g., "*.txt" for only text files in the root)
#         "**/*.txt" will search recursively in subdirectories as well
# - loader_cls: Specify TextLoader to handle the actual file loading
# - loader_kwargs: Pass arguments to the TextLoader, like encoding
loader = DirectoryLoader(
    path=data_directory, 
    glob="**/*.txt",        # Load all .txt files recursively
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=True,     # Optional: Show a progress bar
    use_multithreading=True # Optional: Speed up loading for many files
)

try:
    documents = loader.load()

    # Print the number of documents loaded
    print(f"\nLoaded {len(documents)} documents.")

    if documents:
        # Type of the documents variable
        print(f"\nType of documents list: {type(documents)}")
        print(f"Type of individual document: {type(documents[0])}")

        # Print the first document content (or a snippet)
        print(f"\nFirst document content snippet:\n---\n{documents[0].page_content[:200]}...\n---")

        # Print Document metadata for the first document
        # Metadata will include the 'source' (file path)
        print(f"\nFirst document metadata: {documents[0].metadata}")
        
        # You could also loop through to see all sources:
        print("\nSources of loaded documents:")
        for doc in documents:
            print(f"- {doc.metadata.get('source', 'N/A')}")

    else:
        print("\nNo documents were loaded. Check the directory path and glob pattern.")

except Exception as e:
    print(f"\nAn error occurred during loading: {e}")


