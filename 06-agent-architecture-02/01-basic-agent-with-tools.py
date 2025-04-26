import ast
from typing import Annotated, TypedDict
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
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
    
# Defining Tool - Calculator. Simple calculator that can evaluate mathematical expressions.
# We use ast.literal_eval to safely evaluate the expression.
# ast.literal_eval only allows certain Python literals (strings, numbers, tuples, lists, dicts, booleans, and None).
# It does not allow arbitrary code execution, making it safer than eval.
# Example: "2 + 2" will return 4, "3 * (4 + 5)" will return 27.
@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)

# Adding a search tool to the agent.
search = DuckDuckGoSearchRun()
# Defining the final list of tools to be used in the agent.
tools = [search, calculator]
# Initializing the model 
model = ChatOllama(
    model="qwq",
    temperature=0,
).bind_tools(tools)

# Defining the state of the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Defining a function to print the current messsages in the State
def print_messages(state: State):
    """Prints the messages in the state."""
    print("\n\nCurrent messages in the state:============================\n")
    for message in state["messages"]:
        print(message.content)
    print("===============================================================\n\n")

# Defining the model node 
# This just calls the model with a list of messages and returns the response.
def model_node(state: State) -> State:
    """Calls the model with the messages in the state and returns the response."""
    # Call the model with the messages in the state
    res = model.invoke(state["messages"])
    # Print the messages in the state
    #print_messages(state)
    # Add the response to the messages in the state
    return {"messages": res}    

# Building the graph
builder = StateGraph(State)
builder.add_node("model", model_node) # model_node is a function that takes the state and returns the state
# ToolNode is a prebuilt node that takes a list of tools and returns the result of the tool that was called.
# ToolNode  executes the tool calls requested in the latest AI message found in the state and returns a ToolMessage with the results.
# ToolNode also handles exceptions and errors that occur during tool execution.
# The error message will be added to the ToolMessage which is passed to the model node to be processed by the LLM. 
# Then the LLM will decide what to do next.
builder.add_node("tools", ToolNode(tools)) 
builder.add_edge(START, "model")
# tools_condition is a prebuilt condition that checks if the output message is a tool call.
# tools_condition serves as a conditional edge.
# It looks at the latest AI message in the state and routes to the tools node if there are any tools to execute.
# Otherwise it ends the graph.
builder.add_conditional_edges("model", tools_condition)
# Tool Output is passed to the model node again to further process. 
builder.add_edge("tools", "model")
# This Graph loops between the model node and the tools node. 
# Which means the model is in charge of deciding when to end the computation, which is a key feature of the agent architecture.
# Whenever we want to model a loop in LangGraph, we need to add a conditional edge. 
# That allows to define a stop condition for the Graph 

# The graph is compiled to create a callable object.
graph = builder.compile()

# Using the Graph
# Defining the input to the graph
input = {
    "messages": [
        HumanMessage(
            "How old was the 30th president of the United States when he died?"
        )
    ]
}

# Based on this input, The Agent needs to first identify the 30th president of United States. 
# Then he needs to get date of birth and the date of death of the president.
# Then he needs to calculate the age of the president when he died.
# The agent will use the search tool to get the information about the 30th president of United States.
# The agent will use the calculator tool to calculate the age of the president when he died.

# The graph is called with the input and the output is streamed.
for chunk in graph.stream(input):
    # print('========================')
    # print(chunk)
    # print('========================')
    if "model" in chunk:
        print("\n\nModel Output:\n")
        print(chunk["model"]["messages"])
    if "tools" in chunk:
        print("\n\nTools Output:\n")
        print(chunk["tools"]["messages"])

# Printing the final State of the Graph
# print("\n\nFinal State of the Graph:============================\n")
# print_messages(graph.state)
# print("===============================================================\n\n")


# Generating the graph visualization
graph_filename = "06-agent-architecture-02/01-basic-agent-with-tools.png"
visualize_graph(graph, graph_filename)