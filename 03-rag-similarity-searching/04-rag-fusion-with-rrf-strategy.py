from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain


# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# Load the document, split it into chunks
raw_documents = TextLoader('sample-data/test.txt', encoding='utf-8').load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Create embeddings for the documents
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

db = PGVector.from_documents(
    documents, 
    embeddings_model, 
    connection=connection,
    collection_name="test.txt",
    distance_strategy="cosine",
    use_jsonb=True,
    pre_delete_collection=True
)

# create retriever to retrieve 2 relevant documents
retriever = db.as_retriever(search_kwargs={"k": 2})

prompt_rag_fusion = ChatPromptTemplate.from_template(
    """You are a helpful assistant that generates multiple search queries based on a single input query. \n Generate 4 search queries related to: {question} \n Output only the generated (4 queries):""")


def parse_queries_output(message):
    return message.content.split('\n')


llm = ChatOllama(
    model="gemma3:27b", 
    temperature=0
)
# Definine a chain for generating multiple queries
query_gen = prompt_rag_fusion | llm | parse_queries_output

query = "Who are the key figures in the ancient greek history of philosophy?"

generated_queries = query_gen.invoke(query)

print("generated queries: ")
for i, query in enumerate(generated_queries):
    print(f'{i}. {query}')

print("\n\n")

# we fetch relevant documents for each query and pass them into a function to rerank (that is, reorder according to relevancy) the final list of relevant documents.

def reciprocal_rank_fusion(results: list[list], k=60):
    """reciprocal rank fusion on multiple lists of ranked documents and an optional parameter k used in the RRF formula"""
    # Initialize a dictionary to hold fused scores for each document
    # Documents will be keyed by their contents to ensure uniqueness
    fused_scores = {}
    documents = {}
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            doc_str = doc.page_content
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc
            fused_scores[doc_str] += 1 / (rank + k)
    # sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d: fused_scores[d], reverse=True)
    return [documents[doc_str] for doc_str in reranked_doc_strs]


retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion

result = retrieval_chain.invoke(query)

print("retrieved context using rank fusion: ", result[0].page_content)
print("\n\n")

print("Use model to answer question based on retrieved docs\n")


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)

query = "Who are the some important yet not well known philosophers in the ancient greek history of philosophy?"


@chain
def rag_fusion(input_question):
    # fetch relevant documents
    docs = retrieval_chain.invoke(input_question)  # format prompt
    formatted = prompt.invoke(
        {"context": docs, "question": input_question})  # generate answer
    answer = llm.invoke(formatted)
    return answer


# run
print("Running rag fusion\n")
result = rag_fusion.invoke(query)
print(result.content)
