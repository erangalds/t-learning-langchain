# Note: No OpenAI authentication or .env loading is needed for local Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os # os is still used by OllamaEmbeddings internally sometimes, keep it

# 1. Initialize the Ollama Embeddings model
# Ensure Ollama service is running and has the nomic-embed-text model pulled
# (ollama run nomic-embed-text)
try:
    print("Initializing Ollama embeddings model (nomic-embed-text)...")
    # Specify the model name you have pulled in Ollama
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    print("Ollama embeddings model initialized.")
except Exception as e:
    print(f"Error initializing Ollama embeddings model: {e}")
    print("Please ensure the Ollama service is running and the 'nomic-embed-text' model is available.")
    exit()

# 2. Embed a single query (string)
single_text = "The quick brown fox jumps over the lazy dog."
print(f"\nEmbedding single text: '{single_text}'")

try:
    single_embedding = embeddings_model.embed_query(single_text)

    print(f"Type of single embedding: {type(single_embedding)}")
    print(f"Length of single embedding vector: {len(single_embedding)}")
    print(f"First few dimensions of the embedding: {single_embedding[:5]}...")

except Exception as e:
    print(f"Error embedding single text: {e}")


# 3. Embed a list of documents (strings)
list_of_texts = [
    "LangChain provides tools for building applications with LLMs.",
    "Embeddings represent text as numerical vectors.",
    "Ollama allows running models like nomic-embed-text locally."
]
print(f"\nEmbedding a list of {len(list_of_texts)} texts...")

try:
    bulk_embeddings = embeddings_model.embed_documents(list_of_texts)

    print(f"Type of bulk embeddings result: {type(bulk_embeddings)}")
    print(f"Number of embeddings generated: {len(bulk_embeddings)}")

    if bulk_embeddings:
        print(f"Type of individual embedding in the list: {type(bulk_embeddings[0])}")
        print(f"Length of the first embedding vector: {len(bulk_embeddings[0])}")
        print(f"First few dimensions of the first embedding: {bulk_embeddings[0][:5]}...")

except Exception as e:
    print(f"Error embedding list of texts: {e}")

