from langchain.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document

# --- Configuration ---
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
namespace = "my_docs_namespace"

# Initialize the PGVector vector store  
try:
    vectorstore = PGVector(
        embeddings=embeddings_model,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )
except Exception as e:
    print(f"Error initializing PGVector: {e}")
    exit()

# Initialize the SQLRecordManager
# This will create a table in the database to store the metadata
try:
    record_manager = SQLRecordManager(
        namespace,
        db_url="postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
    )
except Exception as e:
    print(f"Error initializing SQLRecordManager: {e}")
    exit()

# Create the schema if it doesn't exist
try:
    record_manager.create_schema()
except Exception as e:
    print(f"Error creating schema: {e}")
    exit()

# Create documents
docs = [
    Document(page_content='there are cats in the pond', metadata={"id": 1, "source": "cats.txt"}),
    Document(page_content='ducks are also found in the pond', metadata={"id": 2, "source": "ducks.txt"}),
]

# Index the documents
try:
    index_1 = index(
        docs,
        record_manager,
        vectorstore,
        cleanup="incremental",  # prevent duplicate documents
        source_id_key="source",  # use the source field as the source_id
    )

    print("Index attempt 1:", index_1)
except Exception as e:
    print(f"Error during index attempt 1: {e}")

# second time you attempt to index, it will not add the documents again
try:
    index_2 = index(
        docs,
        record_manager,
        vectorstore,
        cleanup="incremental",
        source_id_key="source",
    )

    print("Index attempt 2:", index_2)
except Exception as e:
    print(f"Error during index attempt 2: {e}")

# If we mutate a document, the new version will be written and all old versions sharing the same source will be deleted.

docs[0].page_content = "I just modified this document!"
try:
    index_3 = index(
        docs, 
        record_manager, 
        vectorstore, 
        cleanup="incremental", 
        source_id_key="source"
    )
    print("Index attempt 3:", index_3)
except Exception as e:
    print(f"Error during index attempt 3: {e}")
