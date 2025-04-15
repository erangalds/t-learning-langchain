from langchain_postgres import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


# Define PVector connection string
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain" # "postgresql://postgres:password@localhost:5432/postgres"
# Initialize the Embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Initiave the PGVector vector store
vector_store = PGVector(
    connection=connection,
    collection_name="manually_add_documents",
    distance_strategy="cosine",
    embeddings=embeddings
)

# Define a list of documents to add
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
ids = [doc.metadata["id"] for doc in docs]
# Add documents to PGVector
vector_store.add_documents(docs, ids=ids)


# If you want to delete a document, you can do so by its ID
#vector_store.delete(ids=["3"]) # Deletes documents with IDs 1 and 2
