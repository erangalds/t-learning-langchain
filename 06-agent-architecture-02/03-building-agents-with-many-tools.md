# How to build AI Agents with many tools

This script demonstrates a more advanced agent architecture using LangGraph, designed to handle scenarios where an agent might have access to a large number of tools. Instead of making all tools available to the Language Model (LLM) at all times (which can be inefficient or lead to confusion for the LLM), this script introduces a preliminary step to dynamically select a relevant subset of tools based on the user's query.

Let's break down the code:

1. Imports:

    + `ast`: For `ast.literal_eval` in the calculator tool.
    + `typing.Annotated`, `typing.TypedDict`: For defining the graph's `State`.
    + `langchain_community.tools.DuckDuckGoSearchRun`: The search tool.
    + `langchain_core.documents.Document`: Used to represent tool descriptions when creating the vector store.
    + `langchain_core.messages.HumanMessage`: Represents user input.
    + `langchain_core.tools.tool`: Decorator for creating custom tools.
    + `langchain_core.vectorstores.in_memory.InMemoryVectorStore`: A simple vector store that holds data in memory. Used here to store tool descriptions for retrieval.
    + `langchain_ollama.ChatOllama`, `langchain_ollama.OllamaEmbeddings`:
    + `ChatOllama`: The LLM integration (model "qwq").
    + `OllamaEmbeddings`: Used to generate numerical representations (`embeddings`) of text, specifically for the tool descriptions. The "nomic-embed-text" model is used for this.
    + `langgraph.graph.START`, `langgraph.graph.StateGraph`: Core LangGraph components.
    + `langgraph.graph.message.add_messages`: Helper for updating message lists in the state.
    + `langgraph.prebuilt.ToolNode`, `langgraph.prebuilt.tools_condition`: Pre-built components for tool execution and conditional routing.

2. `visualize_graph` Function:

    ```python
    def visualize_graph(graph, filename="graph.png"):
        # ... (implementation) ...
    ```    

    This is the same utility function seen in the other scripts. It generates a PNG visualization of the compiled LangGraph.

3. Tool Definitions:

    + calculator tool:

        ```python
        @tool
        def calculator(query: str) -> str:
            """A simple calculator tool. Input should be a mathematical expression."""
            return ast.literal_eval(query)
        ```

    + search tool:

        ```python
        search = DuckDuckGoSearchRun()
        ```

    + tools list:

        ```python
        tools = [search, calculator]
        ```

        A list of all available tools.

4. Embeddings and Model Initialization:

    ```python
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    model = ChatOllama(
        model="qwq",
        temperature=0,
    )
    ```

    + `embeddings`: An instance of `OllamaEmbeddings` is created. This will be used to convert the textual descriptions of the tools into vector embeddings.
    + `model`: The `ChatOllama` LLM is initialized. Crucially, note that .`bind_tools(tools)` is NOT called here. The `tools` will be bound to the model dynamically later, only after a relevant subset has been selected.

5. Tool Retriever Setup (Key New Concept):

    ```python
    tools_retriever = InMemoryVectorStore.from_documents(
        [Document(tool.description, metadata={"name": tool.name}) for tool in tools],
        embeddings,
    ).as_retriever()
    ```

    + This is the core of the tool selection mechanism.
    + `[Document(tool.description, metadata={"name": tool.name}) for tool in tools]`:
        + It iterates through the tools list.
        + For each tool, it creates a `langchain_core.documents.Document` object.
        + The `page_content` of the Document is set to `tool.description` (e.g., "A simple calculator tool...").
        + The `metadata` of the Document stores the tool.name (e.g., "calculator", "duckduckgo_search").
    + `InMemoryVectorStore.from_documents(...)`: An in-memory vector store is created using these Document objects and the embeddings model. The vector store will:
        + Use the embeddings model to convert each tool's description into a numerical vector.
        + Store these vectors along with their corresponding documents (and metadata).
    + `.as_retriever()`: This converts the vector store into a "retriever." A retriever is an object that can take a query (a string) and find the most semantically similar documents (in this case, tool descriptions) from the vector store.

6. State Definition (State class):

    ```python
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        selected_tools: list[str]
    ```

    + This defines the graph's state.
    + `messages`: The usual list for conversation history.
    + `selected_tools: list[str]`: This is a new field. It will store a list of names of the tools that have been selected as relevant for the current user query.

7. Node Definitions:

    + model_node function (Modified):

        ```python
        def model_node(state: State) -> State:
            selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
            res = model.bind_tools(selected_tools).invoke(state["messages"])
            return {"messages": res}
        ```    
        + This node now first filters the global tools list to get only those whose names are present in `state["selected_tools"]`.
        + `model.bind_tools(selected_tools)`: It then binds only this selected subset of tools to the LLM.
        + `invoke(state["messages"])`: The LLM is called with the current messages and awareness of only the relevant tools.
    + `select_tools` function (New Node):

        ```python
        def select_tools(state: State) -> State:
            query = state["messages"][-1].content
            tool_docs = tools_retriever.invoke(query)
            return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}
        ```    

        + This node is designed to be one of the first steps in the graph.
        + `query = state["messages"][-1].content`: It extracts the user's latest query.
        + `tool_docs = tools_retriever.invoke(query)`: It uses the `tools_retriever` to find tool descriptions that are semantically similar to the user's query. `tool_docs` will be a list of Document objects.
        + `return {"selected_tools": [doc.metadata["name"] for doc in tool_docs]}`: It extracts the names of these relevant tools from their metadata and updates the `selected_tools` field in the graph's state.

8. Graph Construction:

    ```python
    builder = StateGraph(State)
    builder.add_node("select_tools", select_tools)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(tools)) # ToolNode still needs all tools to execute by name

    builder.add_edge(START, "select_tools")
    builder.add_edge("select_tools", "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    graph = builder.compile()
    ```

    + Nodes:
        + "`select_tools`": The new node for selecting tools.
        + "`model`": The modified model node that uses selected tools.
        + "`tools`": The standard `ToolNode`. It's initialized with the full list of tools because it needs to be able to execute any tool by its name if the LLM (even with a subset) decides to call it.
    + Edges (Defining the flow):
        + `builder.add_edge(START, "select_tools")`: The graph starts by going to the `select_tools` node.
        + `builder.add_edge("select_tools", "model")`: After tools are selected, the graph proceeds to the model node. The model node will now have access to the selected_tools list in the state.
        + `builder.add_conditional_edges("model", tools_condition)`: Same as before. If the model node's output contains tool calls (from the selected tools it was aware of), it goes to the tools node. Otherwise, it typically ends.
        + `builder.add_edge("tools", "model")`: After a tool is executed by the tools node, its output is passed back to the model node for further processing.
    + The flow is: `START` -> `select_tools` (retrieves relevant tool names) -> model (LLM reasons with selected tools) -> [conditional: if model calls a tool -> tools (executes the tool) -> model ... else -> `END`].

9. Graph Execution and Output Streaming:

    ```python
    input = {
        "messages": [
            HumanMessage(
                "How old was the 30th president of the United States when he died?"
            )
        ]
    }

    for chunk in graph.stream(input):
        if "select_tools" in chunk:
            print("\n\nFirst Model Output:\n") # Note: "First Model Output" is a bit of a misnomer here
            print(chunk["select_tools"])
        # ... (rest of the printing logic for "model" and "tools" nodes)
    ```

    + The graph is invoked with an initial `HumanMessage`.
    + The streaming loop prints outputs from different nodes as they execute.
    + The output from the select_tools node will show the list of tool names that were deemed relevant to the input query.

10. Graph Visualization Call:

    ```python
    graph_filename = "06-agent-architecture-02/03-many-tools.png"
    visualize_graph(graph, graph_filename)
    ```
    + Saves a visualization of this new graph structure.

In Summary:

The script 03-many-tools.py introduces a sophisticated agent architecture where:

    1. Tool Discovery/Selection: Before the main LLM reasoning step, a select_tools node uses a retriever (backed by a vector store of tool descriptions) to identify a subset of tools most relevant to the user's query.
    2. Dynamic Tool Binding: The model_node then dynamically binds only this selected subset of tools to the LLM for its reasoning process.
    3. Efficiency and Focus: This approach is beneficial when an agent has access to many tools. It prevents overwhelming the LLM with too many options and helps it focus its reasoning by providing only the most pertinent tools for the task at hand.

This pattern allows for more scalable and efficient agent designs, especially as the number and complexity of available tools grow.