from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
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

# We are going to use two different models configurations for the two different tasks
# 1. Generating SQL queries
# The first model instance will be used to generate SQL queries
# 2. Explaining SQL queries
# The second model instance will be used to explain the generated SQL queries

# Model configuration for generating SQL queries
# The model is set to a low temperature to generate more deterministic outputs
model_low_temp = ChatOllama(
    model="gemma3:27b",
    temperature=0,
)
# Model configuration for explaining SQL queries
# The model is set to a higher temperature to allow for more creative explanations
# This is useful for generating more human-like explanations
model_high_temp = ChatOllama(
    model="gemma3:27b",
    temperature=0.7,
)

# Define the state of the graph
# State will be used to track the state of the Graph execution.
# The state is a dictionary that contains the input and output of the graph
class State(TypedDict):
    # to track conversation history
    messages: Annotated[list, add_messages]
    # input
    user_query: str
    # output
    sql_query: str
    sql_explanation: str

# In this example, the input is not the state, but the user query
class Input(TypedDict):
    user_query: str

# Similarly, the output is not the state, but the SQL query and explanation
# The output is a dictionary that contains the SQL query and explanation
class Output(TypedDict):
    sql_query: str
    sql_explanation: str

# Defining the SystemMessage for the first model
# This message will be used to generate SQL queries
generate_prompt = SystemMessage(
    "You are a helpful data analyst, who generates SQL queries for users based on their questions.The output should be a valid SQL query. "
    "The SQL query should be in the format of a string, and should not contain any additional text or explanations. "
)

# Defining the Node for generating SQL queries - generate_sql
# This node will take the user query as input and generate a SQL query
# The function that will be called when the node is executed
def generate_sql(state: State) -> State:
    # The user query is passed as a message to the model
    user_message = HumanMessage(state["user_query"])
    # The final message to the model is a combination of the system message and the user query and the current conversation history
    # Conversation history is stored in the state under the key "messages"
    messages = [generate_prompt, *state["messages"], user_message]
    # The model is invoked with the final list of messages
    res = model_low_temp.invoke(messages)
    # The model returns a message with the SQL query
    # We are expecting the model to return a valid SQL query and only the SQL query nothing else
    # We return the SQL query as a string as well as the user message and the model response from this function
    return {
        "sql_query": res.content,
        # update conversation history
        "messages": [user_message, res],
    }

# Defining the SystemMessage for the second model configuration
# This message will be used to explain SQL queries
explain_prompt = SystemMessage(
    "You are a helpful data analyst, who explains SQL queries to users."
)

# Defining the Node for explaining SQL queries - explain_sql
# This node will take the SQL query as input and generate an explanation
# The function that will be called when the node is executed
def explain_sql(state: State) -> State:
    # Compiling the final message list to be passed to the model
    # The final message to the model is a combination of the system message and current conversation history. 
    # The current conversation history is saved in the State. We extract that using the messages key. 
    messages = [
        explain_prompt,
        # contains user's query and SQL query from prev step
        *state["messages"],
    ]
    # The model is invoked with the final list of messages
    res = model_high_temp.invoke(messages)
    # The model returns a message with the SQL explanation
    # Function returns the SQL explanation as a string as well as the final message from the model.
    return {
        "sql_explanation": res.content,
        # update conversation history
        "messages": res,
    }

# Defining the Graph
# This Graph the State, a seperate Input and Output formats
builder = StateGraph(State, input=Input, output=Output)
# Adding the nodes to the graph
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)
# Adding the edges to the graph
builder.add_edge(START, "generate_sql") # From Start to generate_sql
builder.add_edge("generate_sql", "explain_sql") # From generate_sql to explain_sql
builder.add_edge("explain_sql", END) # From explain_sql to End

# Compiling the Graph
graph = builder.compile()

# Using the Graph
result = graph.invoke({"user_query": "What is the total sales for each product?"})
# The graph is executed with the input data
# Printing the full result 
print("\nFull result:")
print(result)
# Now we can access the SQL query and explanation from the result
print('\n\nSQL query:\n')
print(result['sql_query'])
print('\n\nSQL explanation:\n')
print(result['sql_explanation'])


# --- Code to Visualize the Generated Graph as a PNG file and save the PNG ---
output_filename = "05-agent-architecture-01/02-chain-of-llm-calls-architecture.png"
visualize_graph(graph, output_filename)

