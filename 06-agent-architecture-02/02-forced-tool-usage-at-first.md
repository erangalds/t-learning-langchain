# What if we wanted to force the AI Agent to use a specific tool at the beginning of the Graph?

This script builds upon the concept of an agent using LangGraph, similar to 01-basic-agent-with-tools.py. However, the key difference and the main purpose of this script is to force the agent to use a specific tool as its very first action, rather than letting the Language Model (LLM) decide whether to use a tool or what tool to use initially.

Here's a step-by-step explanation:

1. Imports:

    + `ast`: The `ast.literal_eval` function is used in the `calculator` tool to safely evaluate mathematical expressions.
    + `typing.Annotated`, `typing.TypedDict`: For defining the structure of the graph's state (State).
    + `uuid.uuid4`: To generate unique IDs for `ToolCall` objects.
    + `langchain_community.tools.DuckDuckGoSearchRun`: A pre-built tool for performing web searches using `DuckDuckGo`.
    + `langchain_core.messages.AIMessage`, `langchain_core.messages.HumanMessage`, `langchain_core.messages.ToolCall`: Core message types. `AIMessage` represents a message from the AI, `HumanMessage` from the user, and `ToolCall` is a structured request from the AI to invoke a specific tool.
    + `langchain_core.tools.tool`: Decorator to easily create custom tools (like the calculator).
    + `langchain_ollama.ChatOllama`: Integration for using `Ollama` to run LLMs locally (model "qwq" in this case).
    + `langgraph.graph.START`, `langgraph.graph.StateGraph:` Core components for building the graph. `START` is the entry point.
    + `langgraph.graph.message.add_messages`: Helper to append new messages to the messages list in the state.
    + `langgraph.prebuilt.ToolNode`, `langgraph.prebuilt.tools_condition`:
        + `ToolNode`: A pre-built node that executes tool calls requested by the LLM.
        + `tools_condition`: A pre-built conditional edge that checks if the last AI message contains tool calls. If yes, it routes to the `ToolNode`; otherwise, it typically ends the graph or routes elsewhere.
    + `pprint`: (Imported but commented out in the streaming loop) For pretty-printing Python data structures, useful for debugging.

2. `visualize_graph` Function:

    ```python
    def visualize_graph(graph, filename="graph.png"):
        # ... (implementation) ...
    ```
    + This is a utility function, identical to the one in the other provided scripts.
    + It takes a compiled LangGraph graph object and a filename.
    + It attempts to generate a PNG image visualizing the graph's structure using `graph.get_graph().draw_mermaid_png()`.
    + It includes error handling for missing dependencies (like langgraph[draw]) needed for visualization.

3. Tool Definitions:

    + calculator tool:

        ```python
        @tool
        def calculator(query: str) -> str:
            """A simple calculator tool. Input should be a mathematical expression."""
            return ast.literal_eval(query)
        ```

        A custom tool that takes a string query (expected to be a mathematical expression like "2 + 2") and returns the evaluated result as a string. `ast.literal_eval` is used for safe evaluation.

    + search tool:
    
        ```python
        search = DuckDuckGoSearchRun()
        ```

        An instance of the pre-built `DuckDuckGoSearchRun` tool.

    + tools list:

        ```python
        tools = [search, calculator]
        ```

        A list containing all available tools for the agent.

4. Model Initialization:

    ```python
    model = ChatOllama(
        model="qwq",
        temperature=0,
    ).bind_tools(tools)
    ```

    + Initializes the `ChatOllama` LLM with the model named "qwq" (you'd need this model pulled in your Ollama setup).
    + `temperature=0` aims for more deterministic, less random outputs.
    + `.bind_tools(tools)`: This crucial step makes the LLM "aware" of the available tools. When the LLM generates a response, it can include special instructions (tool calls) if it thinks a tool can help answer the user's query.

5. State Definition (State class):

    ```python
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    ```    
    
    + Defines the structure of the data that flows through the graph.
    + `messages`: A list that will store the conversation history (human messages, AI messages, tool responses). `Annotated[list, add_messages]` ensures that new messages returned by nodes are appended to this list rather than replacing it.

6. Node Definitions:

    + `model_node` function:

        ```python
        def model_node(state: State) -> State:
            res = model.invoke(state["messages"])
            return {"messages": res}
        ```
        This is the standard LLM interaction node. It takes the current state, extracts the messages, sends them to the LLM (model.invoke), and returns the LLM's response to be added to the state's messages. This node is used after the initial forced tool call.

    + `first_model` function (The Key Component):

        ```python
        def first_model(state: State) -> State:
            query = state["messages"][-1].content
            search_tool_call = ToolCall(
                name="duckduckgo_search", args={"query": query}, id=uuid4().hex
            )
            return {"messages": AIMessage(content="", tool_calls=[search_tool_call])}
        ```   

        + This node is designed to be the first node executed after the `START` node.
        + It takes the current state.
        + `query = state["messages"][-1].content`: It extracts the content of the latest message, which is assumed to be the initial `HumanMessage` (the user's query).
        + `search_tool_call = ToolCall(...)`: Crucially, instead of asking the LLM what to do, it programmatically creates a `ToolCall` object.
            + `name="duckduckgo_search"`: Specifies that the `DuckDuckGoSearchRun` tool should be called.
            + `args={"query": query}`: Provides the user's original query as the argument to the search tool.
            + `id=uuid4().hex`: Assigns a unique ID to this tool call.
        + `return {"messages": AIMessage(content="", tool_calls=[search_tool_call])}`: It returns a new `AIMessage`. This AI message has no textual content (content="") but contains the `search_tool_call` in its tool_calls attribute. This signals to the LangGraph framework that a tool needs to be executed.

7. Graph Construction:

    ```python
    builder = StateGraph(State)

    builder.add_node("first_model", first_model)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "first_model")
    builder.add_edge("first_model", "tools")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    graph = builder.compile()
    ```

    + `builder = StateGraph(State)`: Initializes the graph builder with the defined State.
    + Nodes:
        + `builder.add_node("first_model", first_model)`: Adds the `first_model` function as a node.
        + `builder.add_node("model", model_node)`: Adds the standard `model_node`.
        + `builder.add_node("tools", ToolNode(tools))`: Adds the pre-built `ToolNode`, configured with the available tools. This node will execute any tool calls present in the latest `AIMessage`.
    + Edges (Defining the flow):
        + `builder.add_edge(START, "first_model")`: The graph execution begins at `START` and immediately goes to the `first_model` node.
        + `builder.add_edge("first_model", "tools")`: After `first_model` executes (and generates the forced `ToolCall`), the graph transitions to the tools node. The `ToolNode` will see the `ToolCall` from `first_model` and execute the `duckduckgo_search`.
        + `builder.add_conditional_edges("model", tools_condition)`: After the model node runs (this happens after the first tool call and its result have been processed by the model), this edge checks the `tools_condition`:
            + If the LLM's latest response (from `model_node`) contains further tool calls, it routes back to the tools node.
            + If there are no tool calls, the `tools_condition` typically routes to an end state (implicitly `END` in this setup if no other path is specified for the "else" condition of `tools_condition`).
        + `builder.add_edge("tools", "model")`: After the tools node executes a tool and gets a result (a `ToolMessage`), the graph transitions back to the model node. The `model_node` will then process the original messages plus the tool's output to generate the next response or the final answer.
    + `graph = builder.compile()`: Compiles the defined nodes and edges into an executable graph.

    The flow is: `START` -> `first_model` (forces search) -> tools (executes search) -> model (processes search results and decides next step) -> [conditional: if model calls another tool -> tools -> model ... else -> `END`].

8. Graph Execution and Output Streaming:

    ```python
    input = {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    for chunk in graph.stream(input):
        if "first_model" in chunk:
            print("\n\nFirst Model Output:\n")
            print(chunk["first_model"]["messages"])
        if "model" in chunk:
            print("\n\nModel Output:\n")
            print(chunk["model"]["messages"])
        if "tools" in chunk:
            print("\n\nTools Output:\n")
            print(chunk["tools"]["messages"])
    ```
    + An input dictionary is prepared with the initial `HumanMessage`.
    + `graph.stream(input)`: Executes the graph with the given input and streams the output. For each step (node execution) in the graph, it yields a "chunk" representing the state update from that node.
    + The loop iterates through these chunks and prints the messages produced by each relevant node (`first_model`, model, tools) as they become available. This allows you to see the agent's "thought process" step-by-step.

9. Graph Visualization Call:

    ```python
    graph_filename = "06-agent-architecture-02/02-force-first-tool.png"
    visualize_graph(graph, graph_filename)
    ```

    Finally, it calls the visualize_graph function to generate and save a PNG image of the graph's structure to the specified path.

In Summary:

The script 02-force-first-tool.py demonstrates how to create a LangGraph agent that is explicitly directed to use a specific tool (DuckDuckGo search) with the user's initial query as its very first action. This is achieved by having a custom node (first_model) at the beginning of the graph that programmatically constructs a ToolCall, bypassing an initial LLM decision for the first step. After this forced tool execution, the agent proceeds with its normal LLM-driven reasoning, potentially using other tools or generating a final answer based on the initial tool's output.

This approach is useful when you know that a particular type of query will always benefit from a specific tool upfront, or when you want to guide the agent's initial behavior more directly.