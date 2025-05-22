# Building an AI Agent with Tools usage

This script, /Users/eranga/Documents/my-learning/langchain/l-learning-langchain/06-agent-architecture-02/01-basic-agent-with-tools.py, demonstrates how to build a basic conversational agent using *LangGraph*. This agent can understand user queries and decide to use predefined tools (a calculator and a web search) to help answer those queries.

Let's break down the code step-by-step:

1. Imports:

    + `ast`: Used by the `calculator` *tool* to safely evaluate mathematical expressions.
    + `typing.Annotated`, `TypedDict`: For *type hinting*, especially for defining the structure of the agent's state.
    + `langchain_community.tools.DuckDuckGoSearchRun`: Provides a *tool* for performing web searches using **DuckDuckGo**.
    + `langchain_core.messages.HumanMessage`: Represents a message from a human user.
    + `langchain_core.tools.tool`: A *decorator* to easily create custom *tools*.
    + `langchain_ollama.ChatOllama`: An interface to a local LLM (like "qwq" in this case) running via *Ollama*.
    + `langgraph.graph.START`, `StateGraph`: Core components for building the graph. `START` is a special node indicating the graph's entry point, and `StateGraph` is the class used to define the graph's structure.
    + `langgraph.graph.message.add_messages`: A helper to manage lists of messages in the state, ensuring new messages are appended correctly.
    + `langgraph.prebuilt.ToolNode`, `tools_condition`: Pre-built components for LangGraph. ToolNode handles the execution of tools, and t`ools_condition `is a function that decides the next step based on whether the LLM's last message was a tool call.

2. `visualize_graph` Function:

    + This utility function takes a compiled LangGraph graph and attempts to generate a visual representation of it as a PNG image (e.g., "graph.png").
    + It uses `graph.get_graph().draw_mermaid_png()` to create the image.
    + It includes error handling in case the necessary drawing dependencies (like *langgraph*[draw]) are not installed.

3. Tool Definition (calculator):

    ```python
    @tool
    def calculator(query: str) -> str:
        """A simple calculator tool. Input should be a mathematical expression."""
        return ast.literal_eval(query)
    ```

    + The `@tool` decorator turns the calculator function into a Langchain tool that the agent can use.
    + The docstring is important as it provides a description that the LLM can use to understand what the tool does and when to use it.
    + `ast.literal_eval`(query) safely evaluates a string containing a Python literal or container display (e.g., "2 + 2", "10 * 5"). It's safer than eval() because it only allows simple expressions, preventing arbitrary code execution.

4. Tool and Model Setup:

    ```python
    search = DuckDuckGoSearchRun()
    tools = [search, calculator]
    model = ChatOllama(
        model="qwq",
        temperature=0,
    ).bind_tools(tools)
    ```

    + An instance of `DuckDuckGoSearchRun` is created for web searching.
    + `tools` is a list containing both the search tool and the calculator tool.
    + A `ChatOllama` model is initialized (using a local model named "qwq" with `temperature=0` for more deterministic outputs).
    + Crucially, `.bind_tools(tools)` tells the LLM about the available tools. This allows the LLM to, when appropriate, output a special message indicating it wants to call one of these tools with specific arguments.

5. State Definition (State class):

    ```python
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    ```

    + This defines the structure of the data that will flow through the graph. It's a dictionary-like object.
    + `messages`: This key will hold a list of messages (e.g., `HumanMessage, AIMessage, ToolMessage`). The `Annotated[list, add_messages]` part ensures that when new messages are added to this list, they are appended correctly by LangGraph. This list represents the conversation history and intermediate steps.

6. `print_messages` Function (Utility):

    + A helper function (commented out in the main execution flow) to print the current messages in the state for debugging purposes.

7. `model_node` Function:

    ```python
    def model_node(state: State) -> State:
        """Calls the model with the messages in the state and returns the response."""
        res = model.invoke(state["messages"])
        return {"messages": res}
    ```
    
    + This function defines a "node" in the graph.
    + It takes the current state (which includes the messages list).
    + It calls `model.invoke(state["messages"])`, sending the current conversation history to the LLM.
    + The LLM's response (which could be a direct answer or a request to use a tool) is then returned to update the messages in the state.

8. Building the Graph (StateGraph):

    ```python
    builder = StateGraph(State)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    graph = builder.compile()
    ```

    + `builder = StateGraph(State)`: Initializes a new stateful graph, telling it to use the State structure defined earlier.
    + `builder.add_node("model", model_node)`: Adds a *node* named "model" to the graph, which will execute the model_node function.
    + `builder.add_node("tools", ToolNode(tools))`: Adds another *node* named "*tools*". *ToolNode* is a *pre-built LangGraph* component that automatically handles the execution of any *tool calls* requested by the "*model*" *node*. It takes the list of *tools* as an argument.
    + `builder.add_edge(START, "model")`: Defines the entry point of the *graph*. When the *graph* starts, it will first execute the "*model*" *node*.
    + `builder.add_conditional_edges("model", tools_condition)`: This is a crucial part for agentic behavior. After the *"model" node* runs:
        + `tools_condition` checks the last message from the LLM.
        + If the LLM requested a *tool call*, `tools_condition` directs the graph to the *"tools" node*.
        + If the LLM did not request a *tool call* (meaning it thinks it has a final answer), `tools_condition` will typically lead to an end state (or another path if defined). In this basic setup, if no *tool* is called, the graph effectively stops.
    + `builder.add_edge("tools", "model")`: After the "tools" node executes a tool (e.g., performs a search), its output (a `ToolMessage`) is sent back to the "model" node. This allows the LLM to process the tool's result and decide what to do next (e.g., answer the question, call another tool, or ask for clarification).
    + `graph = builder.compile()`: Compiles the defined graph structure into a runnable Langchain object.

9. Using the Graph:

    ```python
    input = {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    for chunk in graph.stream(input):
        if "model" in chunk:
            print("\n\nModel Output:\n")
            print(chunk["model"]["messages"])
        if "tools" in chunk:
            print("\n\nTools Output:\n")
            print(chunk["tools"]["messages"])
    ```

    + An input dictionary is prepared, containing the initial `HumanMessage`.
    + `graph.stream(input)` executes the graph with the given input. The `stream` method allows you to get results incrementally as each node in the graph finishes its execution.
    + The loop iterates through these chunks of output. Each chunk is a dictionary where the key is the `name` of the node that just ran, and the value is the state after that node's execution.
    + The code then prints the messages from the "model" or "tools" node when they appear in the stream.

10. Agent Behavior for the Example Query: For the query "How old was the 30th president of the United States when he died?":

    + Start -> Model: The `HumanMessage` goes to the `model_node`. The LLM, realizing it doesn't know this information directly but has a search tool, will likely output an `AIMessage` containing a `tool_calls` attribute, requesting to use `duckduckgo_search` with a query like "30th president of the United States death age" or similar.
    + Model -> Tools (via tools_condition): `tools_condition` sees the tool call and routes to the `ToolNode`.
    + Tools Node: The `ToolNode` executes the `duckduckgo_search` tool. The tool returns its findings (e.g., "Calvin Coolidge, the 30th U.S. president, was born on July 4, 1872, and died on January 5, 1933."). This result is wrapped in a `ToolMessage`.
    + Tools -> Model: The `ToolMessage` (containing the search results) is added to the messages list in the state and sent back to the `model_node`.
    + Model Node: The LLM now receives the original question and the search results. It can process this information.
        + If the search result directly gives the age, the LLM might formulate a final answer.
        + If the search result gives birth and death dates, the LLM might:
            + Attempt to calculate the age itself if it's simple.
            + Or, it could decide to use the calculator tool by outputting another AIMessage with a `tool_calls` attribute for the calculator (e.g., calculator("1933 - 1872") or a more precise date difference if it can formulate that). If it calls the calculator, the loop (Model -> Tools -> Model) would repeat.
    + Eventually, the LLM will produce an `AIMessage` without any `tool_calls`, which `tools_condition` will interpret as the final answer, and the graph execution for this input will conclude.

11. Graph Visualization Call:

    ```python
    graph_filename = "06-agent-architecture-02/01-basic-agent-with-tools.png"
    visualize_graph(graph, graph_filename)
    ```
    + Finally, the `visualize_graph` function is called to save a diagram of this agent's structure to a PNG file.


In essence, this script sets up a loop: the LLM (model) gets a chance to respond. If it needs more information or needs to perform a calculation, it can request a tool. The tool runs, and its output is fed back to the LLM. This continues until the LLM believes it can answer the user's query. This is a fundamental pattern for building autonomous agents.