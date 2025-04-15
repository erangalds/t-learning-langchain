import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Define the path for the persistent ChromaDB storage
# Assumes the script is run from the project root or adjusts path accordingly
persist_directory = "chroma-db"
collection_name = "manually_add_documents_chroma"

# Ensure the persist directory exists
os.makedirs(persist_directory, exist_ok=True)
print(f"ChromaDB will be persisted to: {os.path.abspath(persist_directory)}")

# Initialize the Embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Define a list of documents to add (same as the original script)
docs = [
    Document(
        page_content="there are cats in the pond",
        metadata={"id": 1, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="ducks are also found in the pond",
        metadata={"id": 2, "location": "pond", "topic": "animals"},
    ),
    Document(
        page_content="fresh apples are available at the market",
        metadata={"id": 3, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the market also sells fresh oranges",
        metadata={"id": 4, "location": "market", "topic": "food"},
    ),
    Document(
        page_content="the new art exhibit is fascinating",
        metadata={"id": 5, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a sculpture exhibit is also at the museum",
        metadata={"id": 6, "location": "museum", "topic": "art"},
    ),
    Document(
        page_content="a new coffee shop opened on Main Street",
        metadata={"id": 7, "location": "Main Street", "topic": "food"},
    ),
    Document(
        page_content="the book club meets at the library",
        metadata={"id": 8, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="the library hosts a weekly story time for kids",
        metadata={"id": 9, "location": "library", "topic": "reading"},
    ),
    Document(
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"id": 10, "location": "community center", "topic": "classes"},
    ),
]

# Using the metadata id field as the document id
ids = [str(doc.metadata["id"]) for doc in docs] # Chroma requires string IDs

print(f"Attempting to create or load Chroma collection '{collection_name}'...")

# Create a new Chroma vector store from the documents, persisting it locally
# and configuring for cosine distance.
# If the collection already exists with the same signature, it will be loaded.
# If it exists with a different signature, it might raise an error.
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    ids=ids,  # Pass the explicit IDs
    collection_name=collection_name,
    persist_directory=persist_directory,
    # Set collection metadata to configure distance metric for new collections
    collection_metadata={"hnsw:space": "cosine"}
)

print(f"Successfully created/loaded Chroma collection '{collection_name}' with {vector_store._collection.count()} documents.")
print("Vector store initialized and documents added.")

# To load the persistent store later in another script or session:
# vector_store_loaded = Chroma(
#     persist_directory=persist_directory,
#     embedding_function=embeddings,
#     collection_name=collection_name
# )
# print(f"Loaded store has {vector_store_loaded._collection.count()} documents.")

# If you want to delete documents, you can do so by their ID after loading/creating the store:
# try:
#     print("\nAttempting to delete document with ID '3'...")
#     vector_store.delete(ids=["3"])
#     print(f"Deletion successful. Document count: {vector_store._collection.count()}")
#
#     # Verify deletion (optional)
#     retrieved_doc = vector_store.get(ids=["3"])
#     if not retrieved_doc or not retrieved_doc.get('ids'):
#         print("Document with ID '3' confirmed deleted.")
#     else:
#         print("Document with ID '3' still found after deletion attempt.")
#
# except Exception as e:
#      print(f"An error occurred during deletion: {e}")

