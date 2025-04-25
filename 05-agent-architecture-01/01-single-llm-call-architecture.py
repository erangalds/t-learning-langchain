from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

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



# Defining the Model
model = ChatOllama(
    model="gemma3:27b",
    temperature=0,
)

# Defininf the State
# The state is a dictionary that contains the input data
# The state is a TypedDict, which means that it has a fixed
# set of keys and values
class State(TypedDict):
    # Messages have the type "list". The `add_messages`
    # function in the annotation defines how this state should
    # be updated (in this case, it appends new messages to the
    # list, rather than replacing the previous messages)
    messages: Annotated[list, add_messages]

# Defining the Node
# The node is a function that takes the state as input
# The function that will be called when the node is executed
def chatbot(state: State):
    # This node is a very simple one, it just calls the model
    # with the messages in the state and returns the answer
    # The model is a ChatOllama model, so it takes a list of
    # messages as input and returns a message as output
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


# Defining the Graph
builder = StateGraph(State)
# Defining the Nodes
# The "chatbot" node is defined by the function `chatbot`
builder.add_node("chatbot", chatbot)
# Defining the Edges
builder.add_edge(START, "chatbot") # 
builder.add_edge("chatbot", END)

# Compiling the Graph
# The graph is compiled, which means that the nodes and edges
# are connected and the graph is ready to be executed
# The `compile` function returns a graph object that can be
graph = builder.compile()


# Using the Graph
# The graph is executed with the input data
# The input data is a dictionary with a key "messages" that
input = {"messages": [HumanMessage("hi!, I am eranga.")]}
# The input data is passed to the graph
# The `stream` function returns a generator that yields the
# output data as it is generated
for i, chunk in enumerate(graph.stream(input)):
    print(f"Iteration {i}:") # Just Printing the iteration number
    # print("Chunk:")
    # print(chunk)
    if 'chatbot' in chunk:
        # Trying to retrieve the output of the chatbot node
        # The output data is a dictionary with a key "messages"
        # that contains the answer from the model
        
        for message in chunk["chatbot"]["messages"]:
            # The answer is a list of messages, so we need to
            # iterate over the list and print each message
            print(message.content)

# --- Code to Visualize the Generated Graph as a PNG file and save the PNG ---
output_filename = "05-agent-architecture-01/01-single-llm-call-architecture.png"
visualize_graph(graph, output_filename)