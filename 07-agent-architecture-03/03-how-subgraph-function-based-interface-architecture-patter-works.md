# How *SubGraphs* use *function* based interface architecture pattern works

This walks through how to create a parent graph in LangGraph that invokes a subgraph using a regular function node as an intermediary. This "function-based interface" pattern is particularly useful when the *state schema* of the *parent graph* and the *subgraph* are different, requiring explicit transformation of data between them.

Here's a step-by-step explanation:

1. Import necessary modules:

    + `TypedDict` from `typing` is used to define the structure of the state objects.
    + `START`, `StateGraph` from `langgraph.graph` are core components for building graphs.

2. Define State Schemas:

    + `State(TypedDict)`

        ```python
        class State(TypedDict):
            foo: str
        ```

        This defines the *state* for the *parent graph*. It has a single *key* `foo` of type string.

    + `SubgraphState(TypedDict)`

        ```python
        class SubgraphState(TypedDict):
            # none of these keys are shared with the parent graph state
            bar: str
            baz: str
        ```

        This defines the *state* for the *subgraph*. It has two *keys*, `bar` and `baz`, both strings. Crucially, as the comment notes, these *keys* are not directly shared with the parent graph's *state schema*. The `baz` *key* is defined but not actually used as an input to the `subgraph_node` in this specific example; the string "baz" is hardcoded within it.

3. Define the Subgraph:

    + `subgraph_node(state: SubgraphState)`:

        ```python
        def subgraph_node(state: SubgraphState):
            return {"bar": state["bar"] + "baz"}
        ```

        This function is the single node within the subgraph.

        + It takes the SubgraphState as input.
        + It concatenates the string "baz" to the value of state["bar"].
        + It returns a dictionary updating the bar key in the SubgraphState.

    + Subgraph Construction:

        ```python
        subgraph_builder = StateGraph(SubgraphState)
        subgraph_builder.add_node("subgraph_node", subgraph_node)
        subgraph_builder.add_edge(START, "subgraph_node")
        # Additional subgraph setup would go here
        subgraph = subgraph_builder.compile()
        ```

    + A StateGraph is initialized with SubgraphState.
    + subgraph_node is added as a node named "subgraph_node".
    + The graph starts at "subgraph_node".
    + The subgraph is compiled into an executable LangGraph object.

4. Define the Parent Graph Node (Interface Function):

    + node(state: State):

        ```python
        def node(state: State):
            # transform the state to the subgraph state
            response = subgraph.invoke({"bar": state["foo"]})
            # transform response back to the parent state
            return {"foo": response["bar"]}
        ```
        This function is the core of this pattern. It acts as a node in the parent graph and serves as the interface to the subgraph.
            + It takes the parent graph's State (which contains foo) as input.
            + Transformation to Subgraph State: It calls subgraph.invoke().
                + Notice how it manually creates the input dictionary for the subgraph: {"bar": state["foo"]}. It maps the parent's foo value to the subgraph's expected bar key.
            + Transformation from Subgraph Response: The response from subgraph.invoke() will be a SubgraphState dictionary (e.g., {"bar": "hellobaz", "baz": "initial_baz_if_it_were_passed_and_returned"}).
                + It then manually extracts the relevant data from the subgraph's response (response["bar"]) and maps it back to the parent graph's state structure: {"foo": response["bar"]}.

5. Define the Parent Graph:

    + Parent Graph Construction:

        ```python
        builder = StateGraph(State)
        # note that we are using `node` function instead of a compiled subgraph
        builder.add_node("node", node)
        builder.add_edge(START, "node")
        # Additional parent graph setup would go here
        graph = builder.compile()
        ```

    + A StateGraph is initialized with the parent State.
    + The node function (defined above) is added as a node named "node". This is different from the "direct interface" pattern where you'd add subgraph directly.
    + The parent graph starts at this "node".
    + The graph (parent graph) is compiled.

6. Example Usage:
    
```python
initial_state = {"foo": "hello"}
result = graph.invoke(initial_state)
print(
    f"Result: {result}"
)  # Should transform foo->bar, append "baz", then transform bar->foo
```

    + The parent graph is invoked with an initial state {"foo": "hello"}.
    + Flow:
        1. The parent graph starts, calling its "node" (the node function).
        2. Inside node(state={"foo": "hello"}):
            + subgraph.invoke({"bar": "hello"}) is called.
            + The subgraph executes: subgraph_node(state={"bar": "hello"}) runs, returning {"bar": "hello" + "baz"} which is {"bar": "hellobaz"}.
            + The response in the node function becomes {"bar": "hellobaz"}.
            + The node function then returns {"foo": response["bar"]}, which is {"foo": "hellobaz"}.
        3. This return value becomes the final state of the parent graph.
    + The script prints Result: {'foo': 'hellobaz'}.

Key Takeaway & Comparison:

    + Function-Based Interface (This Script): You use this when the parent and subgraph have different state schemas or when you need more complex logic to prepare data for the subgraph or process its results. A regular Python function in the parent graph explicitly calls subgraph.invoke() and handles the data transformations.
    + Direct Interface (like in 02-subgraph-direct-interface-architecture-pattern.py): You use this when the parent and subgraph share some state keys. You can add the compiled subgraph directly as a node to the parent graph (parent_builder.add_node("subgraph_name", subgraph)). LangGraph automatically passes the shared parts of the state to the subgraph and merges its output back.

This script effectively demonstrates the flexibility of LangGraph in composing graphs, allowing for clear separation of concerns and adaptation between different components (graphs) even when their immediate interfaces (state schemas) don't match directly. The node function acts as an adapter.