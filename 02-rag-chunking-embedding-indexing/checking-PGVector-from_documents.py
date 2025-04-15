from langchain_postgres.vectorstores import PGVector
import psycopg # or psycopg-binary

# You might need a dummy connection or valid one to inspect fully
# but often the signature is available without a live connection.
help(PGVector.from_documents)
