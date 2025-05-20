# How *SubGraph direct interface* architecture pattern works

This script demonstrates a *"subgraph direct interface"* architecture pattern using the *LangGraph* library. In this pattern, a *main graph* (the *parent graph*) can include another *graph* (a *subgraph*) as one of its *nodes*. The "*direct interface*" part means that the *parent graph* and the *subgraph* can directly share and modify parts of the overall *state* if their *state* definitions have overlapping *keys*.

Let's break down the code:

1. *State* Definitions:

    + `State(TypedDict)`: This defines the state for the *parent graph*. It has one *key*, `foo`, which is a string.
    + `SubgraphState(TypedDict)`: This defines the *state* for the *subgraph*. It has two *keys*:
        + `foo`: A string. Notice this *key* is shared with the *parent graph's State*.
        + `bar`: A string, which is local to the *subgraph*.

2. *Subgraph* Definition:

    + `subgraph_node(state: SubgraphState)`: This function is a *node* within the *subgraph*. It takes the `SubgraphState` as input.
        + It accesses the `foo` value from the *state* (which could have been set by the parent graph).
        + It appends the string "bar" to the current value of `state["foo"]`.
        + It returns a dictionary `{"foo": ...}`, updating the `foo` *key* in the `SubgraphState`.
    + `subgraph_builder = StateGraph(SubgraphState)`: An instance of `StateGraph` is created, specifically for managing the `SubgraphState`.
    + `subgraph_builder.add_node("subgraph_node", subgraph_node)`: The `subgraph_node` function is added as a *node* named "`subgraph_node`" to this builder.
    + `subgraph_builder.add_edge(START, "subgraph_node")`: This sets "`subgraph_node`" as the entry point for the *subgraph*.
    + `subgraph_builder.add_edge("subgraph_node", END)`: After "`subgraph_node`" executes, the *subgraph* finishes.
    + `subgraph = subgraph_builder.compile()`: The *subgraph* definition is compiled into a *runnable LangGraph* object.

3. Parent Graph Definition:

    + `builder = StateGraph(State)`: An instance of `StateGraph` is created for the *parent graph*, managing the *State*.
    + `builder.add_node("subgraph", subgraph)`: This is the core of the "*direct interface*" pattern. The compiled *subgraph* itself is added as a *node* to the *parent graph*. *LangGraph* will automatically handle passing the relevant parts of the *parent State* to the `SubgraphState` (specifically, the shared `foo` *key*) and updating the *parent State* with any changes made by the *subgraph* to shared *keys*.
    + `builder.add_edge(START, "subgraph")`: The "`subgraph`" *node* (which is our entire compiled *subgraph*) is set as the entry point for the parent graph.
    + `graph = builder.compile()`: The *parent graph* definition is compiled into a *runnable LangGraph* object.

4. Example Usage:

    + `initial_state = {"foo": "hello"}`: The *parent graph* is initialized with `foo` set to "hello".
    + `result = graph.invoke(initial_state)`: The *parent graph* is executed with this *initial state*.
        + When the parent graph's "subgraph" node (our compiled subgraph) is invoked, LangGraph passes the `foo`: "hello" part of the parent's state to the *subgraph*.
        + The `subgraph_node` inside the `subgraph` receives this, appends "bar" to `foo`, so `foo` becomes "hellobar".
        + Since `foo` is a *shared key*, this modification by the *subgraph* directly updates the foo key in the parent graph's state.
        + `print(f"Result: {result}")`: This will print the final *state* of the *parent graph*. The expected output is `{'foo': 'hellobar'}`.

## In essence:

The script shows how you can encapsulate a piece of logic (the subgraph) and seamlessly integrate it into a larger workflow (the parent graph). The "direct interface" comes from the fact that both graphs operate on a state that has shared keys. When the subgraph is invoked as a node in the parent graph, LangGraph automatically maps the shared parts of the state. This is different from a "function-based interface" (seen in 03-subgraph-function-based-interface-architecture-pattern.py) where you'd typically have an explicit function in the parent graph to translate state to and from the subgraph's specific input/output format if the states weren't directly compatible or shared.

This direct sharing simplifies the integration when the subgraph's state is a superset or a compatible subset of the parent graph's state concerning the keys the subgraph operates on.