import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# defining the function to authenticate with OpenAI
def authenticate_with_openai():
    """
    Authenticate with OpenAI using the API key from environment variables.
    """
    # Construct the path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Assumes script is one level down
    dotenv_path = os.path.join(project_root, ".env") # Path to the .env file in project root

    print(f"Attempting to load .env file from: {dotenv_path}")
    # Load environment variables from the .env file
    loaded = load_dotenv(dotenv_path=dotenv_path)
    if not loaded:
        print(f"Warning: .env file not found or not loaded from {dotenv_path}")

    # Load the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Check if the API key is set
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set. Check your .env file.")
    # Optional: You could return the key or just rely on the environment
    # return openai_api_key

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_directory = os.path.join(project_root, "sample-data", "super-heros")

# 1. Authenticate and load API key
try:
    authenticate_with_openai()
    print("OpenAI authentication successful (API key found in environment).")
except ValueError as e:
    print(f"Authentication Error: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during authentication: {e}")
    exit()

# 2. Load documents from the directory
print(f"\nLoading documents from: {data_directory}")
if not os.path.isdir(data_directory):
    print(f"Error: Directory not found at {data_directory}")
    exit()

try:
    loader = DirectoryLoader(
        path=data_directory,
        glob="**/*.txt",        # Load all .txt files recursively
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    if not documents:
        print("No documents loaded. Exiting.")
        exit()

except Exception as e:
    print(f"Error loading documents: {e}")
    exit()

# 3. Split documents into chunks
print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")
if not chunks:
    print("No chunks created. Exiting.")
    exit()

# Prepare chunk content for embedding
chunk_texts = [chunk.page_content for chunk in chunks]

# 4. Initialize the OpenAI Embeddings model
print("\nInitializing embeddings model...")
try:
    # Ensure API key is available for the embeddings class
    # OpenAIEmbeddings reads it from the environment by default
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception as e:
    print(f"Error initializing embeddings model: {e}")
    exit()

# 5. Generate embeddings for the chunks
print(f"Generating embeddings for {len(chunk_texts)} chunks...")
try:
    chunk_embeddings = embeddings_model.embed_documents(chunk_texts)
    print("Embeddings generated successfully.")

    print(f"\nNumber of embedding vectors generated: {len(chunk_embeddings)}")

    if chunk_embeddings:
        # Show info for the first few chunks/embeddings
        num_to_show = min(3, len(chunks)) # Show up to 3
        print(f"\n--- Embedding Info for First {num_to_show} Chunks ---")
        for i in range(num_to_show):
            print(f"\n[Chunk {i+1}]")
            print(f"Source: {chunks[i].metadata.get('source', 'N/A')}")
            # print(f"Content Snippet: '{chunks[i].page_content[:80]}...'") # Optional: show content snippet
            print(f"Embedding Vector Length: {len(chunk_embeddings[i])}")
            print(f"First 5 dimensions: {chunk_embeddings[i][:5]}...")
            print("-" * 20)

except Exception as e:
    print(f"Error generating embeddings: {e}")

