import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres.vectorstores import PGVector

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_directory = os.path.join(project_root, "sample-data", "super-heros")

# Defining the PostgreSQL connection parameters
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

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
print("\nCreating PGVector index...")
try:
    # Create a PGVector index
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embedding_model,
        connection=connection,
        collection_name="super-heros",
        distance_strategy="cosine", # Optional: Use "euclidean" or "dot_product" as well
        # ids=Optional List["id"]=None, # Optional: List of IDs for the documents
        pre_delete_collection=True # Optional: Uncomment to clear existing collection before adding
    )
    print("PGVector index created.")
except Exception as e:
    print(f"Error creating PGVector index: {e}")
    exit()
