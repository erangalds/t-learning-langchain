# Relection Agent Architecture

This script demonstrates a "reflection" architecture pattern using LangGraph, where an AI agent generates content (an essay) and then another part of the agent (or a different agent) reflects on or critiques that content, leading to iterative improvement.

Let's break it down step by step:

1. Imports

```python 
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
```

+ `typing.Annotated, TypedDict`: These are used for *type hinting*. `TypedDict` helps define a *dictionary*-like structure for the state of our graph, and Annotated allows us to add metadata to types, which langgraph uses for `add_messages`.
+ `langchain_core.messages`: This imports different types of messages used in *LangChain* to communicate with language models.
+ `AIMessage`: A *message* from the AI.
+ `BaseMessage`: The *base class* for all *message* types.
+ `HumanMessage`: A *message* from the *human user*.
+ `SystemMessage`: A *message* that sets the *context* or *instructions* for the AI, not typically displayed to the user.
+ `ChatOllamalangchain_ollama.ChatOllama`: This imports the ChatOllama class, which allows you to use local LLMs (like Gemma, Llama, etc.) running via Ollama as your language model.
+ `langgraph.graph`.`END`,`START`,`StateGraph`: These are core components from LangGraph for building stateful, multi-step agentic applications.
+ `StateGraph`: The main class for defining the structure of your agent's workflow as a graph.
+ `START`: A special node name indicating the entry point of the graph.
+ `END`: A special node name indicating a terminal state of the graph.
+ `langgraph.graph.message.add_messages`: This is a helper function that LangGraph uses to automatically append new messages to a list of messages in the state.

2. visualize_graph Function

```python
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
```

This function takes a compiled LangGraph graph object and an optional filename.
It attempts to generate a visual representation of the *graph* using `graph.get_graph().draw_mermaid_png()`. This method internally uses `Mermaid.js` syntax and a *headless browser* (often via `Playwright`) to render the *graph* as a *PNG image*.
The generated *image bytes* are then written to the specified filename.
It includes error handling for `ImportError` (`if langgraph[draw]` dependencies are missing) and other general exceptions that might occur during image generation (e.g., *Playwright/Chromium* issues).

3. Model Initialization

```python
# Initialize chat model
model = ChatOllama(
    model="gemma3:27b",
    temperature=0,    
)
```

An instance of `ChatOllama` is created.

+ `model="gemma3:27b"`: Specifies that the Gemma 3 model with 27 billion parameters, served by a local Ollama instance, should be used. You'd need to have Ollama running and this model pulled (*ollama pull gemma3:27b*).
+ `temperature=0`: This setting controls the randomness of the model's output. A temperature of 0 makes the output more deterministic and focused, which is often preferred for tasks where consistency is important.

4. State Definition

```python
# Define state type
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

A `State class` is defined using `TypedDict`. This *class* represents the shared state that is passed between the *nodes* in our *LangGraph*.
`messages`: `Annotated[list[BaseMessage], add_messages]`: This defines a single key in our state dictionary: messages.
It's a list of BaseMessage objects (which can be `HumanMessage`, `AIMessage`, or `SystemMessage`).
The `Annotated[..., add_messages]` part tells *LangGraph* that whenever a *node* returns a new *list* of *messages* for this *key*, those *messages* should be appended to the existing *list* of *messages* in the *state*, rather than overwriting it. This is crucial for maintaining conversation history.

5. Prompt Definitions

```python
# Define prompts
generate_prompt = SystemMessage(
    "You are an essay assistant tasked with writing excellent 3-paragraph essays."
    " Generate the best essay possible for the user's request."
    " If the user provides critique, respond with a revised version of your previous attempts."
)

reflection_prompt = SystemMessage(
    "You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission."
    " Provide detailed recommendations, including requests for length, depth, style, etc."
)
```

Two `SystemMessage` *object*s are defined. These *messages* provide high-level *instructions* or *personas* for the LLM for different tasks.
+ `generate_prompt`: This *prompt* instructs the LLM to act as an "essay assistant." It specifies the desired output (3-paragraph essay) and how to handle *critique* (revise previous attempts).
+ `reflection_prompt`: This *prompt* instructs the LLM to act as a "teacher grading an essay." It asks for *critique* and detailed recommendations.

6. generate Function (Node)

```python
def generate(state: State) -> State:
    answer = model.invoke([generate_prompt] + state["messages"])
    return {"messages": [answer]}
```

This function defines a *"node"* in our *LangGraph*. It's responsible for generating the essay.
It takes the current state (which is a State dictionary) as input.
`[generate_prompt] + state["messages"]`: It constructs the input for the LLM by prepending the `generate_prompt` to the current *list* of *messages* in the *state*. This gives the LLM the *context* of its *role* and the ongoing conversation (including the initial user request and any previous critiques).
`model.invoke(...)`: Calls the LLM with the combined messages. The LLM's response will be an `AIMessage`.
`return {"messages": [answer]}`: It returns a *dictionary* that updates the *state*. The new `AIMessage` (the generated essay or revision) is wrapped in a list and assigned to the messages key. Because of `add_messages` in the State definition, this new *message* will be appended to the existing *messages list* in the graph's *state*.

7. reflect Function (Node)

```python
def reflect(state: State) -> State:
    # Invert the messages to get the LLM to reflect on its own output
    cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [reflection_prompt, state["messages"][0]] + [
        cls_map[msg.__class__](content=msg.content) for msg in state["messages"][1:]
    ]
    answer = model.invoke(translated)
    # We treat the output of this as human feedback for the generator
    return {"messages": [HumanMessage(content=answer.content)]}
```

This function defines another *node* in the *graph*, responsible for reflecting on or critiquing the generated essay.
`cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}`: This *dictionary* is used to *"invert"* the *roles* of *messages*. The idea here is to make the LLM critique its own previous `AIMessage` as if it were a `HumanMessage` submission.
`translated = [reflection_prompt, state["messages"][0]] + [...]`:
It starts the input for the reflection LLM call with the `reflection_prompt` (to set the "teacher" persona) and the original user request (`state["messages"][0]`, which is assumed to be the first `HumanMessage`).
`[cls_mapmsg.__class__ for msg in state["messages"][1:]]`: This is the clever part. It iterates through the rest of the *messages* in the state (skipping the initial user request).
If a *message* was an *AIMessage* (the essay generated by the *generate node*), it's converted into a `HumanMessage`. This makes the reflection LLM see its own previous output as a "submission."
If a *message* was a `HumanMessage` (which in this loop would be a *critique* from a previous *reflection* step), it's converted into an *AIMessage*. This isn't strictly necessary for the LLM to understand but maintains a consistent pattern of "dialogue" for the reflection task.
`answer = model.invoke(translated)`: The LLM is called with these *"translated" messages* and the `reflection_prompt`. Its output will be an `AIMessage` containing the *critique*.
`return {"messages": [HumanMessage(content=answer.content)]}`: The *critique* (content of the `AIMessage` from the reflection LLM) is then packaged as a new `HumanMessage`. This is because, in the next iteration, the *generate node* expects *critiques* to be *HumanMessages*. This simulates *human feedback* for the *essay generator*.

8. `should_continue` Function (Conditional Edge)

```python
def should_continue(state: State):
    if len(state["messages"]) > 8:
        # End after 3 iterations, each with 2 messages
        return END
    else:
        return "reflect"
```

This function determines the next step after the generate node has run. It's used for conditional routing in the graph.
It checks the total number of messages in the state.
The logic `len(state["messages"]) > 8` implies a *loop* structure:
    1. Initial User Request (1 message)
    2. Generate (adds 1 AI message) -> Total 2
    3. Reflect (adds 1 Human critique) -> Total 3
    4. Generate (adds 1 AI message) -> Total 4
    5. Reflect (adds 1 Human critique) -> Total 5
    6. Generate (adds 1 AI message) -> Total 6
    7. Reflect (adds 1 Human critique) -> Total 7
    8. Generate (adds 1 AI message) -> Total 8 If after a generate call, the message count is greater than 8 (e.g., 9 after the 4th generation, if we count the initial message + 4 generations + 3 reflections), the process stops. Effectively, this allows for roughly:
        1. Initial request
        2. Generation 1
        3. Reflection 1
        4. Generation 2 (revision)
        5. Reflection 2
        5. Generation 3 (revision)
        6. Reflection 3
        7. Generation 4 (revision) -> then stops. 
        So, it's about 3-4 iterations of generate-reflect. The comment "End after 3 iterations, each with 2 messages" is a bit simplified. It's more like an initial message, then 3-4 pairs of (generate, reflect) messages, plus one final generation.
If the condition is met, it returns END, signaling LangGraph to terminate the execution.
Otherwise, it returns the string "reflect", which is the name of the next node to execute.

9. Graph Building

```python
# Build the graph
builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()
```

`builder = StateGraph(State)`: An instance of StateGraph is created, configured to use our defined State type.
`builder.add_node("generate", generate)`: Adds the generate function as a node named "generate".
`builder.add_node("reflect", reflect)`: Adds the reflect function as a node named "reflect".
`builder.add_edge(START, "generate")`: Defines an edge from the special START entry point to the "generate" node. This means when the graph starts, it will first execute the "generate" node.
`builder.add_conditional_edges("generate", should_continue)`: This is crucial for the iterative loop.

It specifies that after the *"generate" node* finishes, the `should_continue` *function* should be called.
The *return value* of `should_continue` (either `END` or "`reflect`") determines the next step. If it's "reflect", the *graph transitions* to the *"reflect" node*. If it's `END`, the *graph terminates*.
`builder.add_edge("reflect", "generate")`: Defines an edge from the "reflect" node back to the "generate" node. This completes the loop: after reflection, the process goes back to generation (for revision).
`graph = builder.compile()`: *Compiles* the defined *graph structure* into an executable *LangGraph* object.

10. Graph Visualization Call

```python
# Generating the graph visualization
graph_filename = "07-agent-architecture-03/01-reflection-architecture-pattern.png"
visualize_graph(graph, graph_filename)
```
A *filename* for the graph image is defined.
The `visualize_graph` function (defined earlier) is called with the compiled graph and the `graph_filename` to save a `PNG` *image* of the *graph's structure*.

11. Example Usage

```python
# Example usage
initial_state = {
    "messages": [
        HumanMessage(
            content="Write an essay about the relevance of 'The Little Prince' today."
        )
    ]
}

# Run the graph
for output in graph.stream(initial_state):
    message_type = "generate" if "generate" in output else "reflect"
    print("\nNew message:", output[message_type]
          ["messages"][-1].content[:], "...")
```

`initial_state`: A *dictionary* defining the starting *state* for the *graph*. It contains a single `HumanMessage` with the essay topic.
`graph.stream(initial_state)`: This executes the graph. The `stream()` method is used because the *graph execution* is *stateful* and produces outputs at each step (each *node* completion). It returns an iterator.
The for loop iterates through the outputs yielded by `graph.stream()`. Each output is a *dictionary* representing the *state* after a *node* has executed. The *key* in this *dictionary* will be the name of the *node* that just ran (e.g., `{"generate": {"messages": [...]}}` or `{"reflect": {"messages": [...]}}`).
`message_type = "generate" if "generate" in output else "reflect"`: Determines which *node* just produced the output.
`print(...)`: Prints the content of the last message added by the most recently executed node. output       `[message_type]["messages"][-1].content[:]` accesses this. The `[:]` and ... are just for concise printing.

## In Summary:
The script sets up a cyclical process:

+ Start: The user provides an essay topic.
+ Generate: An LLM (as an "essay assistant") writes an essay based on the topic and any previous feedback.
+ Should Continue?:
    + If a certain number of iterations have passed, End.
    + Otherwise, proceed to Reflect.
+ Reflect: Another LLM (or the same LLM with a different persona, as a "teacher") critiques the generated essay. This critique is framed as if it's human feedback.
+ Go back to Generate: The essay assistant now uses this new critique to revise its essay.

This loop continues until the should_continue condition decides to end the process, ideally resulting in a progressively improved essay. The `visualize_graph` function helps in understanding this flow by creating a diagram.
