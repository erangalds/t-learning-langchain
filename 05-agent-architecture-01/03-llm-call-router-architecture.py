from typing import Annotated, Literal, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Defining a function to visualize the graph
def visualize_graph(graph, filename="graph.png"):
    """
    Visualizes the graph and saves it as a PNG file.
    """
    # 1. Generate the PNG image data (bytes)
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()

        # 2. Define the filename for the output PNG
        output_filename = filename 

        # 3. Open the file in binary write mode ('wb') and write the bytes
        with open(output_filename, "wb") as f:
            f.write(png_bytes)

        print(f"Graph visualization saved as {output_filename}")

    except ImportError:
        # Handle cases where necessary drawing dependencies might be missing
        print("\n>>>>> To generate the PNG visualization, you might need to install extra dependencies.")
        print(">>>>> Try running: pip install langgraph[draw]")
    except Exception as e:
        # Catch other potential errors during PNG generation
        print(f"\nAn error occurred while generating the PNG: {e}")
        print("Ensure necessary system dependencies (like Playwright/Chromium if used by draw_mermaid_png) are installed.")

    # --- End of PNG generation code ---

# We need an embedding model to convert the documents into vectors 
# and do a similarity search to identify the route to be taken by the router.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# We are going to use two different models configurations for the two different tasks
# First model instance will be using to identify the domain to route the user query to
# and the second model instance will be used to generate the final answer.
# Model configuration for identifying the domain
# The model is set to a low temperature to generate more deterministic outputs
model_low_temp = ChatOllama(model="gemma3:27b", temperature=0)
# Model configuration for generating the final answer
# The model is set to a higher temperature to allow for more creative answers
model_high_temp = ChatOllama(model="gemma3:27b", temperature=0.7)


# Define the state of the graph
class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    user_query: str
    # output
    domain: Literal["records", "insurance"]
    documents: list[Document]
    answer: str

# Define the input and output of the graph
# The input is a dictionary that contains the user query
class Input(TypedDict):
    user_query: str

# The output is a dictionary that contains the answer to the user query
class Output(TypedDict):
    documents: list[Document] # Chunks of documents retrieved
    answer: str


# Sample documents for testing
sample_docs = [
    Document(
        page_content="Patient medical record...", metadata={"domain": "records"}
    ),
    Document(
        page_content="Insurance policy details...", metadata={"domain": "insurance"}
    ),
]

# Initializing two VectorStores for the two domains
# Initialize vector stores
medical_records_store = InMemoryVectorStore.from_documents(sample_docs, embeddings)
medical_records_retriever = medical_records_store.as_retriever()

insurance_faqs_store = InMemoryVectorStore.from_documents(sample_docs, embeddings)
insurance_faqs_retriever = insurance_faqs_store.as_retriever()

# Defining the SystemMessage for the router
# This message will be used to identify the domain to route the user query to
# The router will decide which domain to route the user query to
router_prompt = SystemMessage(
    """You need to decide which domain to route the user query to. You have two domains to choose from:
- records: contains medical records of the patient, such as diagnosis, treatment, and prescriptions.
- insurance: contains frequently asked questions about insurance policies, claims, and coverage.

Output only the domain name."""
)

# Defining the Node for routing the user query
# This node will take the user query as input and route it to the appropriate domain
def router_node(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    # The final message to the model is a combination of the system message and the user query and the current conversation history
    # Conversation history is stored in the state under the key "messages"
    messages = [router_prompt, *state["messages"], user_message]
    # The model is invoked with the final list of messages
    res = model_low_temp.invoke(messages)
    # We expecte the model to return a domain name and only the domain name nothing else
    # We return the domain name as a string as well as the user message and the model response from this function
    return {
        "domain": res.content,
        # update conversation history
        "messages": [user_message, res],
    }

# Defining the Node pick_retriever
# The function to pick the retriever based on the domain
def pick_retriever( state: State) -> Literal["retrieve_medical_records", "retrieve_insurance_faqs"]:
    if state["domain"] == "records":
        return "retrieve_medical_records"
    else:
        return "retrieve_insurance_faqs"

# Defining the Node for retrieving medical records
# The function will take the user query as input and retrieve the relevant documents 
def retrieve_medical_records(state: State) -> State:
    documents = medical_records_retriever.invoke(state["user_query"])
    return {
        "documents": documents,
    }

# Defining the Node for retrieving insurance FAQs
# The function will take the user query as input and retrieve the relevant documents
def retrieve_insurance_faqs(state: State) -> State:
    documents = insurance_faqs_retriever.invoke(state["user_query"])
    return {
        "documents": documents,
    }

# Defining the SystemMessage for the Node that generates the final answer for medical records domain
medical_records_prompt = SystemMessage(
    "You are a helpful medical chatbot, who answers questions based on the patient's medical records, such as diagnosis, treatment, and prescriptions."
)
# Defining the SystemMessage for the Node that generates the final answer for insurance FAQs domain
insurance_faqs_prompt = SystemMessage(
    "You are a helpful medical insurance chatbot, who answers frequently asked questions about insurance policies, claims, and coverage."
)

# Defining the Node for generating the final answer

def generate_answer(state: State) -> State:
    if state["domain"] == "records":
        prompt = medical_records_prompt
    else:
        prompt = insurance_faqs_prompt
    # The final message to the model is a combination of the system message and the user query and the current conversation history
    # and the documents retrieved from based on the user query. 
    messages = [
        prompt,
        *state["messages"],
        HumanMessage(f"Documents: {state['documents']}"),
    ]
    # The model is invoked with the final list of messages
    res = model_high_temp.invoke(messages)
    return {
        "answer": res.content,
        # update conversation history
        "messages": res,
    }

# Defining the Graph
builder = StateGraph(State, input=Input, output=Output)
# Defining the Nodes
# The "router" node is defined by the function `router_node`
builder.add_node("router", router_node)
builder.add_node("retrieve_medical_records", retrieve_medical_records)
builder.add_node("retrieve_insurance_faqs", retrieve_insurance_faqs)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "router")
# We have a edge based on a condition. 
builder.add_conditional_edges("router", pick_retriever)
builder.add_edge("retrieve_medical_records", "generate_answer")
builder.add_edge("retrieve_insurance_faqs", "generate_answer")
builder.add_edge("generate_answer", END)
# Compiling the Graph
graph = builder.compile()

# Using the Graph
input = {"user_query": "Am I covered for COVID-19 treatment?"}
for i, chunk in enumerate(graph.stream(input)):
    #print(f"Iteration {i}:")
    if "router" in chunk:
        print("\n\nRouter Output:\n")
        print(chunk["router"]["domain"])
    if "generate_answer" in chunk:
        print("\n\nFinal Output:\n")
        print(chunk["generate_answer"]["answer"])

# --- Code to Visualize the Generated Graph as a PNG file and save the PNG ---
output_filename = "05-agent-architecture-01/03-llm-call-router-architecture.png"
visualize_graph(graph, output_filename)
