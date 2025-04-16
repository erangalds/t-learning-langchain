# Import necessary components from typing and langchain libraries
from typing import Annotated, TypedDict # Used for defining the structure of our state
from langchain_core.messages import HumanMessage, AIMessage # Represents messages from human/AI
from langchain_ollama import ChatOllama # The specific language model we'll use (Ollama with gemma3)
from langgraph.graph import StateGraph, START, END, add_messages # Core LangGraph components for building the graph
from langgraph.checkpoint.memory import MemorySaver # Not used here

## Graph Components
# `StateGraph` is a specialized graph structure that allows us to define a flow of
# State : The Data Received from outside the application, modified and produced by the application while its running. 
# `State` is a dictionary-like structure that holds the state of the graph.
# Nodes: Functions that process the state. Basically, python functions. 
# Nodes receive the current state as input and can return and update that state.
# Edges: connections between nodes that define the flow of execution.
# Edges can be directed (one-way / fixed) or undirected (two-way / conditional).

# --- Deeper dive into TypedDict ---
# `typing.TypedDict` allows us to define a dictionary type with a fixed set of keys,
# where each key is expected to hold a value of a specific type.
# It's useful for:
#   1. Readability: Clearly documents the expected structure of dictionary-like objects.
#   2. Static Analysis: Tools like MyPy can check if dictionaries used as 'State'
#      actually conform to this structure (e.g., have the 'messages' key with the correct type).
#   3. LangGraph Integration: LangGraph uses this definition to understand and manage
#      the data (state) flowing through the graph. It expects nodes to receive and
#      return dictionaries that conform to this 'State' structure.
#
# At runtime, an object conforming to 'State' is still just a standard Python dictionary.
# TypedDict primarily provides benefits during development and type checking.
#
# Example of a valid 'State' dictionary:
# valid_state = {'messages': [HumanMessage(content='Hello')]}
#
# Example of an invalid 'State' dictionary (would cause issues or type errors):
# invalid_state_1 = {}  # Missing the 'messages' key
# invalid_state_2 = {'messages': "just a string"} # 'messages' value is not a list
# invalid_state_3 = {'message': []} # Key name is wrong ('message' instead of 'messages')
# --- End of TypedDict explanation ---
class State(TypedDict):
    """State for the state graph."""
    # 'messages' will hold a list of chat messages (like HumanMessage, AIMessage)
    # 'Annotated' adds metadata. Here, 'add_messages' tells LangGraph how to update
    # the state when new messages are added. It means that when we add new messages to
    # this list: new messages are appended rather than replacing the old ones.
    # This is important for maintaining the conversation history.
    
    messages: Annotated[list, add_messages]

# Create an instance of StateGraph. We pass our 'State' definition so the graph
# knows what kind of data structure to manage and validate against.
builder = StateGraph(State)

# Initialize the ChatOllama language model
model = ChatOllama(
    model="gemma3:27b", # Specify the Ollama model name
    temperature=0 # Set temperature to 0 for more deterministic, less random responses
)

# Define a function that represents a 'node' in our graph.
# The type hint `state: State` indicates this function expects a dictionary
# conforming to the 'State' TypedDict structure as input.
def chatbot(state: State):
    # We can safely access `state['messages']` because the 'State' TypedDict
    # guarantees this key exists if the input state is valid.
    answer = model.invoke(
        state['messages']
    )
    # This function must return a dictionary that is a valid *update*
    # for the 'State'. Here, it provides the 'messages' key, matching the definition.
    return {'messages': [answer]}

# Add the 'chatbot' function as a node to our graph builder.
# We give it a name 'chatbot'. When the graph reaches this node, it will execute the 'chatbot' function.
builder.add_node('chatbot', chatbot)

# Define the flow (edges) of the graph.
# Add an edge from the special 'START' point to our 'chatbot' node.
# This means when the graph starts, it will first go to the 'chatbot' node.
builder.add_edge(START, 'chatbot')
# Add an edge from the 'chatbot' node to the special 'END' point.
# This means after the 'chatbot' node finishes, the graph execution stops.
builder.add_edge('chatbot', END)

# Compile the graph definition into an executable graph object.
# This finalizes the structure we've defined with nodes and edges.
graph = builder.compile()

# Prepare the initial input for the graph.
# This dictionary MUST conform to the 'State' TypedDict structure.
input_data = { # Renamed 'input' to 'input_data' to avoid shadowing built-in 'input'
    'messages': [HumanMessage(content='hi!')] # Used 'content=' for clarity
}
# Run the graph with the prepared input.
# LangGraph executes the flow: START -> chatbot -> END
# The 'chatbot' node receives the input, calls the LLM, and updates the state.
output = graph.invoke(input_data)

# Print the final state of the graph after execution.
# The 'output' dictionary will also conform to the 'State' TypedDict structure.
print(f'Output: \n{output}\n')

# --- Code to Visualize the Generated Graph as a PNG file and save the PNG ---

# 1. Generate the PNG image data (bytes)
try:
    png_bytes = graph.get_graph().draw_mermaid_png()

    # 2. Define the filename for the output PNG
    output_filename = "04-using-langraph-to-add-memory-chat/02-state_graph_visualization.png"

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

