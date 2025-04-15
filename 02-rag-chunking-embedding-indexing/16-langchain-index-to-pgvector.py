from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter 
import uuid
import os 
from dotenv import load_dotenv # Import load_dotenv

# defining the function to authenticate with OpenAI
def authenticate_with_openai():
    """
    Authenticate with OpenAI using the API key from environment variables.
    """
    dotenv_path = "../.env" # Path to the .env file
    # Load environment variables from the .env file
    load_dotenv(dotenv_path=dotenv_path) 
    # Load the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Check if the API key is set
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")

# 1. Authenticate and load API key
try:
    authenticate_with_openai()
    print("OpenAI authentication successful.")
except ValueError as e:
    print(f"Authentication Error: {e}")
    print("Please ensure your OPENAI_API_KEY is set in the correct .env file.")
    exit()
except FileNotFoundError:
    print("Error: Could not find the .env file for authentication.")
    print("Please ensure '../.env' exists relative to common_utils/authenticate-with-openai.py")
    exit()

# --- Configuration ---
# Assuming this script is in a directory one level below the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_file = os.path.join(project_root, "sample-data", "test.txt")

# Defining the PostgreSQL connection parameters
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain" # "postgresql://postgres:password@localhost:5432/postgres"

# 1. Load a single document from the file
print(f"\nLoading document from: {data_file}")
if not os.path.isfile(data_file):
    print(f"Error: File not found at {data_file}")
    exit()
try:
    loader = TextLoader(data_file, encoding='utf-8')
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

# Define Embedding Model
print("\nInitializing embeddings model...")
try:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embeddings model initialized.")
except Exception as e:  
    print(f"Error initializing embeddings model: {e}")
    exit()
# 3. Create a PGVector index
print("\nCreating PGVector index...")
try:
    # Create a PGVector index
    # When this method is called, it will create a new collection in the database
    # and insert the documents into it.
    # langchain will create two tables in the database:
    # langchain_pg_collection - which stores the collection name and collection id
    # langchain_pg_embeddings - which stores the embeddings and metadata and collection id.
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        connection=connection,
        collection_name="langchain_test",
        distance_strategy="cosine",
        # ids=Optional List["id"]=None, # Optional: List of IDs for the documents
        pre_delete_collection=True # Optional: Uncomment to clear existing collection before adding
    )
    print("PGVector index created.")
except Exception as e:
    print(f"Error creating PGVector index: {e}")
    exit()  


