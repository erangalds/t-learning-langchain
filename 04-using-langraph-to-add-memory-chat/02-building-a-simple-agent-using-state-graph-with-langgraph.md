# Building a simple AI Agent using *StateGraph* with *LangGraph*

This script demonstrates how to build a very simple conversational AI (a chatbot) using the LangGraph library. LangGraph allows you to define a series of steps (nodes) and the connections between them (edges) to create more complex AI applications than a single Large Language Model (LLM) call.

Let's break down the code:

1. Imports:

    + `typing.Annotated`, `typing.TypedDict`: These are Python's built-in typing tools.
        + `TypedDict` is used to define a dictionary-like structure called State with specific keys and value types. This helps in making the code more readable and allows for static type checking.
        + `Annotated` is used to add extra information to type hints. Here, it's used with `add_messages` from LangGraph.
    + `langchain_core.messages.HumanMessage`, `langchain_core.messages.AIMessage`: These classes represent messages from a human user and an AI assistant, respectively. They are standard message types in the Langchain ecosystem.
    + `langchain_ollama.ChatOllama`: This is the Langchain integration for using Ollama, which allows you to run various open-source LLMs locally. The script uses the "gemma3:27b" model.
    + `langgraph.graph.StateGraph`, `langgraph.graph.START`, `langgraph.graph.END`, `langgraph.graph.add_messages:` These are core components from LangGraph.
        + `StateGraph`: The main class for building stateful graphs.
        + `START`, `END`: Special nodes representing the beginning and end of the graph's execution.
        + add_messages: A special function used with Annotated to tell LangGraph how to update a list of messages in the state (it appends new messages instead of replacing the whole list).
    + langgraph.checkpoint.memory.MemorySaver: This import is present but commented as "Not used here". In other scripts (like 03-add-persistent-memory.py), it's used to save the graph's state, allowing for conversation memory across multiple interactions.

2. Graph State Definition (State class):

```python
class State(TypedDict):
    """State for the state graph."""
    messages: Annotated[list, add_messages]
```

    + This defines the structure of the data that will flow through your graph. It's a dictionary that must have a key named messages.
    + The value associated with messages must be a list.
    + `Annotated[list, add_messages]` tells LangGraph that when a node returns a new list of messages for the `messages` key, these new messages should be appended to the existing list in the state, rather than overwriting it. This is crucial for building up a conversation history.

3. Graph Builder Initialization:

    ```python
    builder = StateGraph(State)
    ```
    + An instance of StateGraph is created. It's initialized with the State definition, so the graph knows what kind of data to expect and manage.

4. Language Model Initialization:

    ```python
    model = ChatOllama(
        model="gemma3:27b",
        temperature=0
    )
    ```

    + This sets up the LLM that will power the chatbot.
    + `model="gemma3:27b"` specifies the particular Ollama model to use.
    + `temperature=0` makes the model's responses more deterministic and less random. Higher temperatures lead to more creative but potentially less factual or coherent outputs.

5. Chatbot Node Definition (chatbot function):

    ```python
    def chatbot(state: State):
        answer = model.invoke(
            state['messages']
        )
        return {'messages': [answer]}
    ```

    + This function defines a "node" in your graph. A node is a unit of computation.
    + It takes the current state (which is a dictionary conforming to the State TypedDict) as input.
    + `state['messages']` accesses the list of messages from the current state.
    + `model.invoke(state['messages'])` sends these messages to the Ollama LLM and gets a response (an AIMessage object).
    + It returns a dictionary `{'messages': [answer]}`. This dictionary represents an update to the state. Because messages is annotated with add_messages, this new `AIMessage` (the answer) will be appended to the existing list of messages in the graph's state.

6. Adding Nodes and Edges to the Graph:

    ```python
    builder.add_node('chatbot', chatbot)
    builder.add_edge(START, 'chatbot')
    builder.add_edge('chatbot', END)
    ```

    + `builder.add_node('chatbot', chatbot)`: Registers the chatbot function as a node named "chatbot" in the graph.
    + `builder.add_edge(START, 'chatbot')`: Defines a directed connection (an edge) from the special START node to the "chatbot" node. This means when the graph execution begins, it will first execute the "chatbot" node.
    + `builder.add_edge('chatbot', END)`: Defines an edge from the "chatbot" node to the special END node. This means after the "chatbot" node finishes, the graph execution will terminate.
    + This creates a very simple linear graph: START -> chatbot -> END.

7. Compiling the Graph:

    ```python
    graph = builder.compile()
    ```
    + This takes all the defined nodes and edges and compiles them into an executable graph object.

8. Preparing Input and Invoking the Graph:

    ```python
    input_data = {
        'messages': [HumanMessage(content='hi!')]
    }
    output = graph.invoke(input_data)
    ```

    + input_data is the initial state provided to the graph. It must conform to the State TypedDict (i.e., have a messages key with a list of messages). Here, it starts with a single HumanMessage.
    + `graph.invoke(input_data)` runs the graph.
        + The `input_data` becomes the initial state.
        + The graph follows the edges: `START` leads to the "`chatbot`" node.
        + The `chatbot` function is called with the current state (containing "hi!").
        + The LLM generates a response.
        + The chatbot function returns `{'messages': [AIMessage_from_LLM]}`.
        + `add_messages` ensures this AI message is appended to the messages list in the state.
        + The graph then follows the edge from "chatbot" to END, and execution finishes.
    + output will contain the final state of the graph after execution. In this simple case, it will be `{'messages': [HumanMessage(content='hi!'), AIMessage(content='<LLM_response>')]}`.

9. Printing the Output:

    ```python
    print(f'Output: \n{output}\n')
    ```
    + This displays the final state, showing the initial human message and the AI's response.

10. Graph Visualization:

    ```python
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        # ... (code to save png_bytes to a file) ...
    except ImportError:
        # ... (error handling for missing dependencies) ...
    except Exception as e:
        # ... (general error handling) ...
    ```    

    + This section attempts to generate a visual representation of the graph as a PNG image using Mermaid syntax.
    + `graph.get_graph().draw_mermaid_png()` is a LangGraph utility that creates this visualization.
    + The try-except block handles potential errors, such as missing optional dependencies required for drawing (like playwright or pygraphviz). If these are missing, it prints a helpful message suggesting how to install them (pip install langgraph[draw]).
    + The generated image (02-state_graph_visualization.png) would show a simple diagram: [START] --> chatbot --> [END].

In essence, this script does the following:

+ Defines a data structure (State) to hold a list of conversation messages.
+ Sets up an LLM (ChatOllama).
+ Creates a simple graph with one main step: a chatbot function that takes the current messages, gets a response from the LLM, and adds the LLM's response back to the list of messages.
+ Runs the graph with an initial human message ("hi!").
+ Prints the conversation history (the initial message and the AI's reply).
+ Tries to save a visual diagram of this simple graph.

This script is a foundational example. More complex LangGraph applications can have multiple nodes, conditional edges (deciding which node to go to next based on the current state), and more sophisticated state management, including persistent memory as shown in 03-add-persistent-memory.py.