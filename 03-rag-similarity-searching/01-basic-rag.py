from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.exceptions import LangChainException, LangChainError
from langchain_postgres.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# Configuring the PGVector vector connection string
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# Initialize the Ollama embeddings model
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
except LangChainException as e:
    print(f"Error initializing OllamaEmbeddings: {e}")
    exit()

# Initialize the PGVector vector store
try:
    vectorstore = PGVector(
        connection=connection,
        collection_name="test.txt",
        embeddings=embeddings,
        distance_strategy="cosine",
        use_jsonb=True,
        pre_delete_collection=True,  # Delete the collection if it exists
    )
except Exception as e:
    print(f"Error initializing PGVector: {e}")
    exit()

# Load the document
try:
    raw_documents = TextLoader("sample-data/test.txt", encoding= "utf-8").load()
except Exception as e:
    print(f"Error loading document: {e}")
    exit()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

docs = text_splitter.split_documents(raw_documents)
# Index the documents
index = vectorstore.add_documents(docs)
# Initiate a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# User query
query = "Who are the key figures in the ancient greek history of philosophy?"

# Fetch relevant documents
docs = retriever.invoke(query)

# Define a Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", """Answer the question based on the following context
    
    contex: 
    {context}
     
    Question: 
    {question}"""),
])
# Initialize the ChatOllama model   
llm = None
try:
    llm = ChatOllama(model="gemma3:27b", temperature=0)
except ImportError:
    print("Error: langchain_ollama not installed. Please install it using 'pip install langchain-ollama'")
    exit()
except LangChainError as e:
    print(f"Error initializing ChatOllama: {e}")
    exit()


llm_chain = prompt | llm
response = None

# Run the chain with the retrieved documents and the user query
try:
    response = llm_chain.invoke(
        {
            "context": docs,
            "question": query
        }
    )
    if response:
        print(response.content)
    else:
        print("No response received from the LLM chain.")

except Exception as e:
    print(f"Error invoking chain: {e}")
    exit()
    
print("\n\n")
# Run again but this time encapsulate the logic for efficiency

# @chain decorator transforms this function into a LangChain runnable,
# making it compatible with LangChain's chain operations and pipeline

print("Running again but this time encapsulate the logic for efficiency\n")


@chain
def qa(input):
    try:
        # fetch relevant documents
        try:
            docs = retriever.invoke(input)
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return None
        # check if documents are retrieved
        if not docs:
            print("No relevant documents found.")
            return None
        # format prompt
        try:
            formatted = prompt.invoke({"context": docs, "question": input})
            # generate answer
            answer = llm.invoke(formatted)
            return answer
        except Exception as e:
            print(f"Error invoking prompt or llm: {e}")
            return None
    except Exception as e:
        print(f"Error in qa chain: {e}")
        return None

# Invoke the qa chain with the user query
try:
    result = qa.invoke(query)
    print(result.content)
except Exception as e:
    print(f"Error invoking qa chain: {e}")


