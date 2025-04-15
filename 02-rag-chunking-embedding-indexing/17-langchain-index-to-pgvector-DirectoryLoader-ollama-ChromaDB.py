import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_directory = os.path.join(project_root, "sample-data", "super-heros")
chroma_db_directory = os.path.join(project_root, "chroma-db")

# Defining the ChromaDB collection name
collection_name = "super-heros"

# 1. Load documents from the directory
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
    # Load documents
    print(f"Attempting to load files from: {os.path.abspath(data_directory)}")  # Added for debugging path issues
    # Load documents
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
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
# Split documents into chunks
chunks = text_splitter.split_documents(documents)
# If not chunks then we exit
if not chunks:
    print("No chunks created. Exiting.")
    exit()
print(f"Split into {len(chunks)} chunks.")

# 3. Initialize the Ollama Embeddings model
print("\nInitializing Ollama embeddings model (nomic-embed-text)...")
try:
    # Ensure Ollama
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    print("Ollama embeddings model initialized.")
except Exception as e:
    print(f"Error initializing Ollama embeddings model: {e}")
    exit()
# 4. Create a PGVector index
print("\nCreating ChromaDB index...")
try:
    # Create a ChromaDB index
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=chroma_db_directory,
        collection_name=collection_name,
    )
    # Persist the ChromaDB index
    vectorstore.persist()
    print(f"ChromaDB index created and persisted to {chroma_db_directory}.")

except Exception as e:
    print(f"Error creating ChromaDB index: {e}")
    exit()
