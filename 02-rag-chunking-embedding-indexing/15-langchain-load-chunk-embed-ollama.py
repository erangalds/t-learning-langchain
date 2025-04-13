import os
# Note: No OpenAI authentication or .env loading is needed for local Ollama
# Note: No sys import needed if only used for auth path manipulation
from langchain_community.embeddings import OllamaEmbeddings # Import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
# Assuming this script is in a directory one level below the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_directory = os.path.join(project_root, "sample-data", "super-heros")

# 1. Load documents from the directory (No authentication step needed)
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

# 2. Split documents into chunks
print("\nSplitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Keep chunk size reasonable for the model context
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

# 3. Initialize the Ollama Embeddings model
print("\nInitializing Ollama embeddings model (nomic-embed-text)...")
try:
    # Ensure Ollama service is running and has the nomic-embed-text model pulled
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    print("Ollama embeddings model initialized.")
except Exception as e:
    print(f"Error initializing Ollama embeddings model: {e}")
    print("Please ensure the Ollama service is running and the 'nomic-embed-text' model is available.")
    exit()

# 4. Generate embeddings for the chunks
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

