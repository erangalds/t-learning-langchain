import ast
from typing import Annotated, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

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

@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
# We don't bind the tools here as we did earlier
model = ChatOllama(
    model="qwq",
    temperature=0,
)

tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={"name": tool.name}) for tool in tools],
    embeddings,
).as_retriever()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]


def model_node(state: State) -> State:
    selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
    # Bind the tools to the model before invoking the model with inputs
    res = model.bind_tools(selected_tools).invoke(state["messages"])
    return {"messages": res}

# Defining a separate Node as the initial node to get the list of selected tools based on the user input
def select_tools(state: State) -> State:
    # User Query is added to the list of messages from the input variable
    # Therefore, when that's added, the last message in the messages list is the user query
    query = state["messages"][-1].content
    # Getting the list of tools details from the inmemory VectorStore. 
    tool_docs = tools_retriever.invoke(query)
    # Adding those Tool names as a list and adding that to the selected_tools key in the State variable
    return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}

# Defining the Graph
builder = StateGraph(State)
builder.add_node("select_tools", select_tools)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
# First Node is the select_tools Node. Which selects a list of relevant tools to use.
builder.add_edge(START, "select_tools")
builder.add_edge("select_tools", "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()

# Using the Graph
input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

for chunk in graph.stream(input):
    # print('\n\n================= Using PPrint =====================')
    # pprint.pprint(chunk, indent=4)
    # print('\n\n======================================================')
    # print('======================================================\n\n')
    if "select_tools" in chunk:
        print("\n\nFirst Model Output:\n")
        print(chunk["select_tools"])
    if "model" in chunk:
        print("\n\nModel Output:\n")
        print(chunk["model"]["messages"])
    if "tools" in chunk:
        print("\n\nTools Output:\n")
        print(chunk["tools"]["messages"])


# --- Code to Visualize the Generated Graph as a PNG file and save the PNG ---
graph_filename = "06-agent-architecture-02/03-many-tools.png"
visualize_graph(graph, graph_filename)