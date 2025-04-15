
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres.vectorstores import PGVector
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain, RunnablePassthrough


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

# Query starts with irrelevant information before asking the relevant question
query = 'Today I woke up and brushed my teeth, then I sat down to read the news. But then I forgot the food on the cooker. Who are some key figures in the ancient greek history of philosophy?'

# fetch relevant documents
docs = retriever.invoke(query)
# Print the relevant documents retrieved
print("Relevant documents retrieved:")
for doc in docs:
    print(doc.metadata)
    print(doc.page_content)

print("\n\n")

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: {context} Question: {question} """
)
llm = ChatOllama(
    model="gemma3:27b", 
    temperature=0
)

# Run again but this time encapsulate the logic for efficiency
# @chain decorator transforms this function into a LangChain runnable,
# making it compatible with LangChain's chain operations and pipeline

@chain
def qa(input):
    # fetch relevant documents
    docs = retriever.invoke(input)
    # Print the relevant documents retrieved
    print("Relevant documents retrieved:")
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content)
    
    print("\n\n")
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer


# run it
result = qa.invoke(query)
print(result.content)

print("\nRewrite the query to improve accuracy\n")

rewrite_prompt = ChatPromptTemplate.from_template(
    """Provide a better search query for web search engine to answer the given question, end the queries with ’**’ Only provide one rewritten query and only that as the answer. Question: {x} Answer:""")


def parse_rewriter_output(message):
    return message.content.strip('"').strip("**")


rewriter = rewrite_prompt | llm | parse_rewriter_output 


@chain
def qa_rrr(input):
    # Print the original query
    print("Original query: \n", input)
    # rewrite the query
    new_query = rewriter.invoke(input)
    print("Rewritten query: \n", new_query)
    # fetch relevant documents
    docs = retriever.invoke(new_query)
    # Print the relevant documents retrieved
    print("Relevant documents retrieved:")
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content)
    print("\n\n")
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer


print("\nCall model again with rewritten query\n")

# call model again with rewritten query
result = qa_rrr.invoke(query)
print(result.content)
