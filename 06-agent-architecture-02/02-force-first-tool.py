import ast
from typing import Annotated, TypedDict
from uuid import uuid4
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import pprint

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
model = ChatOllama(
    model="qwq",
    temperature=0,
).bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: State) -> State:
    res = model.invoke(state["messages"])
    return {"messages": res}

# This is the Node which we are going to setup as the fist Node
# Purpose of this Node is to force a toolcall
# In this case it to force to do a web search, using the user given query. 
def first_model(state: State) -> State:
    # User Query is added to the list of messages from the input variable
    # Therefore, when that's added, the last message in the messages list is the user query
    query = state["messages"][-1].content
    # Defining a ToolCall, in this case for a duckduckgo_search, with the user query.
    search_tool_call = ToolCall(
        name="duckduckgo_search", args={"query": query}, id=uuid4().hex
    )
    # The search_tool_call contains the output of the tool call. 
    # print(f'\n\nForced Search Tool Call: \n\n{search_tool_call}\n\n')
    # ai_message = AIMessage(content="", tool_calls=[search_tool_call])
    # print(f'\n\nAI Message: \n\n{ai_message}\n\n')

    return {"messages": AIMessage(content="", tool_calls=[search_tool_call])}


builder = StateGraph(State)

builder.add_node("first_model", first_model)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
# First Node is the Forced_Tool_Calling Node
builder.add_edge(START, "first_model")
builder.add_edge("first_model", "tools")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()

# Example usage
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
    if "first_model" in chunk:
        print("\n\nFirst Model Output:\n")
        print(chunk["first_model"]["messages"])
    if "model" in chunk:
        print("\n\nModel Output:\n")
        print(chunk["model"]["messages"])
    if "tools" in chunk:
        print("\n\nTools Output:\n")
        print(chunk["tools"]["messages"])


# --- Code to Visualize the Generated Graph as a PNG file and save the PNG ---
graph_filename = "06-agent-architecture-02/02-force-first-tool.png"
visualize_graph(graph, graph_filename)
