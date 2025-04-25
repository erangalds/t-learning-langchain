import ast
from typing import Annotated, TypedDict
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]
model = ChatOllama(
    model="gemma3:27b",
    temperature=0,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: State) -> State:
    res = model.invoke(state["messages"])
    return {"messages": res}


builder = StateGraph(State)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
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
    if "model" in chunk:
        print(chunk["model"]["messages"].content)

# --- Code to Visualize the Generated Graph as a PNG file and save the PNG ---

# 1. Generate the PNG image data (bytes)
try:
    png_bytes = graph.get_graph().draw_mermaid_png()

    # 2. Define the filename for the output PNG
    output_filename = "06-agent-architecture-02/01-basic-agent-with-tools.png"

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