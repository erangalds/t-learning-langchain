# Single LLM Call Agent Architecture Pattern

This script demonstrates the most basic form of a conversational agent using the LangGraph library. It sets up a simple graph with a single node that interacts with a Large Language Model (LLM) to generate responses.

Let's break down the code step-by-step:

1. Imports:

    + `typing.Annotated`, `typing.TypedDict`: These are Python's built-in typing tools.
        + `TypedDict` is used to define a dictionary-like structure called State with specific keys and value types. This improves code readability and enables static type checking.
        + `Annotated` is used to add extra information to type hints. Here, it's used with `add_messages` from LangGraph.
    + `langgraph.graph.StateGraph`, `langgraph.graph.START`, `langgraph.graph.END`: Core components from LangGraph for building stateful graphs.
        + `StateGraph`: The main class for defining the structure and flow of your agent.
        + `START`, `END`: Special nodes representing the beginning and end of the graph's execution.
    + `langgraph.graph.message.add_messages`: A helper function used with Annotated to specify how a list of messages in the state should be updated. It appends new messages to the existing list, which is crucial for maintaining conversation history.
    + `langchain_ollama.ChatOllama`: This is the Langchain integration for using `Ollama`, which allows you to run various open-source LLMs locally. The script uses the "gemma3:27b" model.
    + `langchain_core.messages.HumanMessage`: This class represents a message from a human user.

2. visualize_graph Function:

    ```python
    def visualize_graph(graph, filename="graph.png"):
        # ... (implementation) ...
    ```    
    
    + This is a utility function designed to create a visual representation of the LangGraph.
    + It takes a compiled graph object and an optional `filename` as input.
    + `graph.get_graph().draw_mermaid_png()`: This LangGraph method attempts to generate a PNG image of the graph using Mermaid syntax (a diagramming language).
    + The generated image data (bytes) is then written to the specified filename.
    + It includes try-except blocks to handle potential errors:
        + `ImportError`: Catches errors if optional dependencies needed for drawing (like `playwright` or `pygraphviz`, installable via pip install `langgraph[draw]`) are missing.
        + `Exception`: Catches other general errors that might occur during image generation.

3. Model Initialization:

    ```python
    model = ChatOllama(
        model="gemma3:27b",
        temperature=0,
    )
    ```

    + This line sets up the LLM that will power the chatbot.
    + `ChatOllama(...)`: Creates an instance of the Ollama chat model client.
    + `model="gemma3:27b"`: Specifies that the "gemma3:27b" model (which you'd need to have pulled and running via Ollama) should be used.
    + `temperature=0`: This parameter controls the randomness of the LLM's output. A temperature of 0 makes the output more deterministic and focused, meaning it's more likely to produce the same or very similar responses to the same input. Higher temperatures lead to more creative or varied, but potentially less predictable, outputs.

4. State Definition (State class):

    ```python
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    ```    

    + This defines the structure of the data that will be passed between nodes in your graph. It's the "memory" or "context" of the conversation.
    + State is a `TypedDict`, meaning it's a dictionary with a predefined set of keys and associated types.
    + `messages`: `Annotated[list, add_messages]`:
        + This declares that the state dictionary will have a key named `messages`.
        + The value associated with `messages` must be a list (intended to store conversation messages like HumanMessage and AIMessage).
        + `Annotated[list, add_messages]` is a special `LangGraph` instruction. It tells the graph that when a node returns a new list of messages for the messages key, these new messages should be appended to the existing list in the state, rather than replacing the entire list. This is how conversation history is built up.

5. Node Definition (chatbot function):

    ```python
    def chatbot(state: State):
        answer = model.invoke(state["messages"])
        return {"messages": [answer]}
    ```

    + This function defines a "node" in your graph. A node represents a unit of computation or a step in your agent's process.
    + It takes the current state (which is a dictionary conforming to the `State` `TypedDict`) as input.
    + `state["messages"]` accesses the list of messages (the conversation history so far) from the current state.
    + `model.invoke(state["messages"])`: This is where the LLM is called. The current conversation history is passed to the `ChatOllama model`. The model processes these messages and generates a response (which will be an `AIMessage` object).
    + `return {"messages": [answer]}`: The function returns a dictionary. This dictionary represents an update to the state.
        + The key is "messages", matching the key in the State definition.
        + The value is [answer], a list containing the new `AIMessage` from the LLM.
        + Because of the `Annotated[list, add_messages]` in the State definition, this new `AIMessage` will be appended to the existing messages list in the graph's state.

6. Graph Construction:

    ```python
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    ```

    + `builder = StateGraph(State)`: An instance of `StateGraph` is created. It's initialized with the State definition, so the graph knows what kind of data to expect and manage.
    + `builder.add_node("chatbot", chatbot)`: This registers the `chatbot` function as a node named "chatbot" within the graph.
    + `builder.add_edge(START, "chatbot")`: This defines a directed connection (an edge) from the special `START `node to the "chatbot" node. This means that when the graph execution begins, it will first execute the "chatbot" node.
    + `builder.add_edge("chatbot", END)`: This defines an edge from the "chatbot" node to the special `END` node. This means that after the "chatbot" node finishes its execution, the graph's process will terminate.
    + This creates a very simple linear graph: `START` -> `chatbot` -> `END`.

7. Compiling the Graph:

    ```python
    graph = builder.compile()
    ```

    + This takes all the defined nodes and edges and compiles them into an executable graph object. The graph object can now be run.

8. Using the Graph (Execution and Output Streaming):

    ```python
    input_data = {"messages": [HumanMessage("hi!, I am eranga.")]}
    for i, chunk in enumerate(graph.stream(input_data)):
        print(f"Iteration {i}:")
        if 'chatbot' in chunk:
            for message in chunk["chatbot"]["messages"]:
                print(message.content)
    ```

    + `input_data = {"messages": [HumanMessage("hi!, I am eranga.")]}`: This dictionary is the initial state provided to the graph. It must conform to the State TypedDict (i.e., have a messages key with a list of messages). Here, the conversation starts with a single `HumanMessage`.
    + `graph.stream(input_data)`: This executes the graph with the given `input_data.` The `.stream()` method is used here, which means the graph will yield results incrementally as each node completes. For a simple graph like this, it might not seem very different from `.invoke()`, but for more complex graphs with multiple steps, streaming allows you to see intermediate results.
    + The for loop iterates through the "chunks" yielded by `graph.stream()`. Each chunk is a dictionary representing the output of a node that just ran.
    + `if 'chatbot' in chunk:`: Checks if the current chunk contains output from the "chatbot" node.
    + `for message in chunk["chatbot"]["messages"]:`: If it's from the "chatbot" node, it iterates through the messages returned by that node (in this case, it will be a single AIMessage).
    + `print(message.content)`: Prints the textual content of the AI's response.

9. Graph Visualization Call:

    ```python
    output_filename = "05-agent-architecture-01/01-single-llm-call-architecture.png"
    visualize_graph(graph, output_filename)
    ```

    + Finally, the visualize_graph function is called with the compiled graph and a desired output_filename. This will attempt to save a PNG image of the graph's structure to the specified path. The image would show a simple flow: [START] --> `chatbot` --> [END].

## In essence, this script does the following:

1. Defines a state structure to hold a list of conversation messages.
2. Sets up an LLM (Ollama's "gemma3:27b").
3. Creates a graph with one primary step (the chatbot node):
    + This node takes the current conversation history.
    + Sends it to the LLM to get a response.
    + Adds the LLM's response back to the conversation history.
4. Runs the graph with an initial human message ("hi!, I am eranga.").
5. Streams and prints the AI's response.
6. Saves a visual diagram of this very simple graph.

This script serves as a "hello world" for LangGraph, illustrating the fundamental concepts of state, nodes, edges, and graph execution in a minimal agent architecture.
