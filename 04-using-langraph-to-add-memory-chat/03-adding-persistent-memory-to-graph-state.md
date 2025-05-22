# How to add persistent memory to graph State

This script demonstrates how to add persistent memory to a LangGraph conversational agent. It builds upon a simple chatbot structure (similar to what's in 02-state-graph.py) but introduces MemorySaver to allow the conversation's state (the history of messages) to be remembered across multiple interactions, provided those interactions are associated with the same "thread" or session.

Let's break down the code:

1. Imports:

    + `typing.Annotated`, `typing.TypedDict`: Standard Python typing tools for defining structured data.
    + `langchain_core.messages.HumanMessage`: Represents a message from a human user.
    + `langchain_ollama.ChatOllama`: The Langchain integration for using Ollama to run LLMs locally (model "gemma3:27b").
    + `langgraph.graph.StateGraph`, `langgraph.graph.START`, `langgraph.graph.END`, `langgraph.graph.add_messages:` Core LangGraph components for building the graph and managing message history.
    + `langgraph.checkpoint.memory.MemorySaver`: This is the key import for adding memory. It's a checkpointer that saves the graph's state in memory.

2. State Definition (State class):

    ```python
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    ```

    + This is identical to the state definition in simpler examples like 02-state-graph.py.
    + It defines that the graph's state will be a dictionary with a `messages` key.
    + The value of `messages` is a list, and `Annotated[list, add_messages]` ensures that new `messages` are appended to this list, maintaining the conversation history.

3. Graph Builder and Model Initialization:

    ```python
    builder = StateGraph(State)

    model = ChatOllama(
        model="gemma3:27b",
        temperature=0,
    )
    ```

    + A `StateGraph` instance is created, configured with the State definition.
    + The `ChatOllama` model is initialized, just like in previous examples.

4. Chatbot Node Definition (chatbot function):

    ```python
    def chatbot(state: State):
        answer = model.invoke(state["messages"])
        return {"messages": [answer]}
    ```
    
    + This function defines the single operational node in the graph.
    + It takes the current state (which includes the messages history).
    + It invokes the LLM with these messages.
    + It returns the LLM's response, which will be appended to the messages list in the state.

5. Adding Nodes and Edges:

    ```python
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    ```

    This sets up a simple linear graph: `START` -> `chatbot` -> `END`.

6. Compiling the Graph with MemorySaver (The Core of Persistence):

    ```python
    graph = builder.compile(checkpointer=MemorySaver())
    ```

    + This is the crucial difference from a stateless graph.
    + When `builder.compile()` is called, the checkpointer argument is provided with an instance of `MemorySaver()`.
    + What `MemorySaver` does:
        + It acts as a storage mechanism for the graph's state.
        + After each step (or at configurable points), LangGraph will save the current state of the conversation (specifically, the State dictionary) using this `checkpointer`.
        + `MemorySaver` stores this state in Python's memory. This means the state persists as long as the Python script is running. For persistence across script runs or different processes, other `checkpointers` (like SqliteSaver for database storage) would be needed.
        + When the graph is invoked with a configurable dictionary containing a thread_id, the `checkpointer` first tries to load any previously saved state for that thread_id. If found, the graph resumes from that state.

7. Visualization Code:

    ```python
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        # ... (code to save png_bytes to a file) ...
        output_filename = "04-using-langraph-to-add-memory-chat/03-add-persistent-memory-graph-visualization.png"
        # ...
    except ImportError:
        # ...
    except Exception as e:
        # ...
    ```

    This section is standard in these examples. It attempts to generate a PNG image visualizing the graph's structure. The structure itself (`START` -> `chatbot` -> `END`) is the same as a stateless graph; the `MemorySaver` affects runtime behavior, not the static graph topology.

8. Configuring a Thread and Invoking the Graph with Persistence:

    ```python
    # Configure thread
    thread1 = {"configurable": {"thread_id": "1"}}

    # Run with persistence
    result_1 = graph.invoke({"messages": [HumanMessage("hi, my name is Jack!")]}, thread1)
    print('\n\nResult 1:\n\n')
    print(result_1)

    result_2 = graph.invoke({"messages": [HumanMessage("what is my name?")]}, thread1)
    print('\n\nResult 2:\n\n')
    print(result_2)
    ```

    + `thread1 = {"configurable": {"thread_id": "1"}}`: This dictionary is crucial for using the checkpointer.
        + The configurable key is a standard LangGraph way to pass runtime configuration.
        + `thread_id`: This identifier tells the MemorySaver which conversation "session" or "thread" this interaction belongs to. All interactions with the same thread_id will share the same persisted state.
    + First Invocation (result_1):
        + `graph.invoke({"messages": [HumanMessage("hi, my name is Jack!")]}, thread1)`
        + The graph is called with an initial message "hi, my name is Jack!".
        + Since this is the first call for thread_id: "1", MemorySaver has no prior state.
        + The chatbot node processes this message, the LLM responds.
        + The final state (containing the human message and the AI's reply) is saved by MemorySaver associated with thread_id: "1".
        + result_1 will contain `{'messages': [HumanMessage(content='hi, my name is Jack!'), AIMessage(content='<AI_reply_to_Jack>')]}`.
    + Second Invocation (result_2):
        + `graph.invoke({"messages": [HumanMessage("what is my name?")]}, thread1)`
        + The graph is called again, importantly, with the same thread1 configuration.
        + `MemorySaver`looks up thread_id: "1" and finds the state saved from the previous interaction. This state (containing "hi, my name is Jack!" and the AI's first reply) is loaded as the starting point for this invocation.
        + The new `HumanMessage("what is my name?")` is appended to this loaded message history (due to add_messages).
        + The chatbot node receives the entire conversation history so far:
            1. "hi, my name is Jack!"
            2. AI's reply to Jack
            3. "what is my name?"
        + The LLM can now use the context from the first interaction to answer "what is my name?".
        + The final state (now with four messages) is saved again by MemorySaver for thread_id: "1".
        + result_2 will contain all four messages, and the AI's last response should correctly identify the name as "Jack".

9. Getting the State:

    ```python
    print('\n\nState:\n\n')
    print(graph.get_state(thread1))
    ```

    + `graph.get_state(thread1)`: This method allows you to retrieve the current persisted state for a given thread_id from the `checkpointer`.
    + It will print the State dictionary, which should contain the complete list of four messages exchanged during the two invocations for thread_id: "1".

In Summary:

The script 03-add-persistent-memory.py demonstrates a fundamental way to give a LangGraph agent memory within a single Python process's lifetime. By compiling the graph with MemorySaver() and using thread_id in the configurable options during invocation:

+ The graph can remember past interactions within the same "thread."
+ The LLM receives a growing history of messages, allowing for more contextually aware and coherent conversations.
+ This is a step up from stateless graphs where each invocation is independent and has no memory of previous turns.

This is a foundational concept for building more sophisticated chatbots that can maintain context over multiple turns of a conversation.