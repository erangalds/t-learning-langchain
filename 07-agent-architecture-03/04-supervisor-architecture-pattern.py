from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, MessagesState, START
from pydantic import BaseModel
import os
from dotenv import load_dotenv # Import load_dotenv

# Defining the function to visualize the graph
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

# defining the function to authenticate with OpenAI
def authenticate_with_openai():
    """
    Authenticate with OpenAI using the API key from environment variables.
    """
    dotenv_path = ".env" # Path to the .env file
    # Load environment variables from the .env file
    load_dotenv(dotenv_path=dotenv_path) 
    # Load the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Check if the API key is set
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")

# Authenticate with OpenAI
authenticate_with_openai()# This function sets the OpenAI API key from an environment variable



class SupervisorDecision(BaseModel):
    next: Literal["researcher", "coder", "FINISH"]


# Initialize model
model = ChatOllama(
    model="gemma3:27b",
    temperature=0,
)
# model = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
# )
supervisor_model = model.with_structured_output(SupervisorDecision)
llm = ChatOllama(
    model="gemma3:27b",
    temperature=0,
)
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
# )

# Define available agents
agents = ["researcher", "coder"]

# Define system prompts
system_prompt_part_1 = f"""You are a supervisor tasked with managing a conversation between the  
following workers: {agents}. Given the following user request,  
respond with the worker to act next. Each worker will perform a  
task and respond with their results and status. When finished,  
respond with FINISH."""

system_prompt_part_2 = f"""Given the conversation above, 
carefully check whether the users request has been satisfied by the AI response. 
if AI response is statisfied then send output as "FINISH" otherwise select one of: {", ".join(agents)} as output."""


def supervisor(state):
    print(f'This is the supervisor function')
    # Printing Current State
    print(f'Current State:\n{state}\n')
    messages = [
        ("system", system_prompt_part_1),
        *state["messages"],
        ("system", system_prompt_part_2),
    ]
    decision_obj = supervisor_model.invoke(messages)
    print(f"Decision: {decision_obj.next}")

    return {"next": decision_obj.next }


# Define agent state
class AgentState(MessagesState):
    next: Literal["researcher", "coder", "FINISH"]


# Define agent functions
def researcher(state: AgentState):
    print(f'This is the researcher function')
    # Checking the messages 
    content = state["messages"][0].content
    print(content)
    # state["messages"][0].content
    # In a real implementation, this would do research tasks
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a research assistant. Analyze the request and provide relevant information.",
            },
            {"role": "user", "content": content},
        ]
    )
    print(f"Researcher response:\n {response.content}\n")
    return {"messages": [response]}


def coder(state: AgentState):
    # Checking the messages
    print(f'This is the coder function')
    content = state["messages"][0].content
    print(content)
    # state["messages"][0].content 
    # In a real implementation, this would write code
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You are a coding assistant. Implement the requested functionality.",
            },
            {"role": "user", "content": content },
        ]
    )
    print(f"Coder response:\n {response.content}\n")
    return {"messages": [response]}


# Build the graph
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

builder.add_edge(START, "supervisor")
# Route to one of the agents or exit based on the supervisor's decision
builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

graph = builder.compile()

# --- Code to Visualize the Generated Graph as a PNG file and save the PNG ---
graph_filename = "07-agent-architecture-03/04-supervisor-architecture-pattern.png"
visualize_graph(graph, graph_filename)

# Example usage
initial_state = {
    "messages": [
        {
            "role": "user",
            "content": "I need help to open an excel file in python and import the data in the sheet named Sheet1 and save it to a csv file named Data.csv.",
        }
    ],
    "next": "supervisor",
}

for output in graph.stream(initial_state):
    print(f'Printing Output:\n{output}\n')
    
    # print(f"\nStep decision: {output.get('next', 'N/A')}")
    # if output.get("messages"):
    #     print(f"Response: {output['messages'][-1].content[:100]}...")
