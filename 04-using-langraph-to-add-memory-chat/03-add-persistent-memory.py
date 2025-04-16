from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

model = ChatOllama(
    model="gemma3:27b",
    temperature=0,
)

# Define a function that represents a 'node' in our graph.
# The type hint `state: State` indicates this function expects a dictionary
def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Add persistence with MemorySaver
# This allows the graph to save and restore its state across invocations.
# The MemorySaver will store the state in memory, allowing for persistence.
# This will save the state of the graph in memory, allowing for persistence.
# Each interaction with the graph will update the state, and the state will be saved in memory.
# This means that the graph will remember the state between invocations.
graph = builder.compile(checkpointer=MemorySaver())

# --- Code to Visualize the Generated Graph as a PNG file and save the PNG ---

# 1. Generate the PNG image data (bytes)
try:
    png_bytes = graph.get_graph().draw_mermaid_png()

    # 2. Define the filename for the output PNG
    output_filename = "04-using-langraph-to-add-memory-chat/03-add-persistent-memory-graph-visualization.png"

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


# Configure thread
thread1 = {"configurable": {"thread_id": "1"}}

# Run with persistence
result_1 = graph.invoke({"messages": [HumanMessage("hi, my name is Jack!")]}, thread1)
# Print the result
print('\n\nResult 1:\n\n')
print(result_1)

result_2 = graph.invoke({"messages": [HumanMessage("what is my name?")]}, thread1)
# Print the result
print('\n\nResult 2:\n\n')
print(result_2)

# Get state
print('\n\nState:\n\n')
# Get the state of the graph after the second invocation
print(graph.get_state(thread1))
