from langchain_openai import OpenAIEmbeddings
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


# 2. Initialize the OpenAI Embeddings model
# By default, it uses "text-embedding-ada-002" but you can specify others
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small") # Example using a newer model

# 3. Embed a single query (string)
single_text = "The quick brown fox jumps over the lazy dog."
print(f"\nEmbedding single text: '{single_text}'")

try:
    single_embedding = embeddings_model.embed_query(single_text)

    print(f"Type of single embedding: {type(single_embedding)}")
    print(f"Length of single embedding vector: {len(single_embedding)}")
    print(f"First few dimensions of the embedding: {single_embedding[:5]}...") # Uncomment to see part of the vector

except Exception as e:
    print(f"Error embedding single text: {e}")


# 4. Embed a list of documents (strings)
list_of_texts = [
    "LangChain provides tools for building applications with LLMs.",
    "Embeddings represent text as numerical vectors.",
    "OpenAI offers powerful language and embedding models."
]
print(f"\nEmbedding a list of {len(list_of_texts)} texts...")

try:
    bulk_embeddings = embeddings_model.embed_documents(list_of_texts)

    print(f"Type of bulk embeddings result: {type(bulk_embeddings)}")
    print(f"Number of embeddings generated: {len(bulk_embeddings)}")

    if bulk_embeddings:
        print(f"Type of individual embedding in the list: {type(bulk_embeddings[0])}")
        print(f"Length of the first embedding vector: {len(bulk_embeddings[0])}")
        print(f"First few dimensions of the first embedding: {bulk_embeddings[0][:5]}...") # Uncomment to see part of the vector

except Exception as e:
    print(f"Error embedding list of texts: {e}")

