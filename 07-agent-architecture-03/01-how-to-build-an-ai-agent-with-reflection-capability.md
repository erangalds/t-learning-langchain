# How to build an *AI Agent* with *reflection* capability using LangGraph

This script demonstrates a "reflection" architecture pattern using the LangGraph library. The core idea is to have an AI agent (or a part of an agent) generate some output, and then another part (or the same agent in a different mode) reflect on or critique that output. This feedback is then used to improve the next generation, creating an iterative refinement loop.

Let's break down the code step-by-step:

1. Imports:

    + typing.Annotated, typing.TypedDict: For defining the structure of the state.
    + langchain_core.messages: Imports various message types (AIMessage, BaseMessage, HumanMessage, SystemMessage) which are fundamental for structuring conversations with LLMs.
    + langchain_ollama.ChatOllama: The LLM used for generation and reflection. In this case, it's configured to use a local Ollama model (gemma3:27b).
    + langgraph.graph.END, langgraph.graph.START, langgraph.graph.StateGraph: Core components from LangGraph for building stateful graphs.
    + langgraph.graph.message.add_messages: A helper function to easily append new messages to a list of messages in the state.

2. visualize_graph Function:

    This is a utility function to generate a PNG image of the graph's structure using graph.get_graph().draw_mermaid_png().
    It includes error handling in case the necessary drawing dependencies (langgraph[draw]) are not installed. This function is helpful for understanding the flow of the agent.

3. Model Initialization:

    ```python
    model = ChatOllama(
        model="gemma3:27b",
        temperature=0,
    )
    ```
    + An instance of ChatOllama is created.
    + model="gemma3:27b" specifies the local LLM to use.
    + temperature=0 makes the model's output more deterministic and less random, which is often preferred for tasks requiring consistency.

4. State Definition:

    ```python
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    ```

    + State defines the data structure that will be passed between nodes in the graph.
    + It contains a single key, messages, which is a list of BaseMessage objects (e.g., HumanMessage, AIMessage).
    + Annotated[..., add_messages] tells LangGraph that when a node returns a new list of messages for this key, these new messages should be appended to the existing list rather than replacing it.

5. Prompts:

    + generate_prompt: A SystemMessage that instructs the LLM to act as an "essay assistant" and generate a 3-paragraph essay. It also tells the LLM to revise its attempts if critique is provided.
    + reflection_prompt: A SystemMessage that instructs the LLM to act as a "teacher grading an essay." It's asked to provide critique and recommendations.

6. Graph Node Construction: generate Function (Node):

    ```python
    def generate(state: State) -> State:
        answer = model.invoke([generate_prompt] + state["messages"])
        return {"messages": [answer]}
    ```
    This function represents a node in the graph responsible for generating content.
    It takes the current state (which includes the history of messages).
    It prepends the generate_prompt to the existing messages and sends them to the model.invoke().
    The LLM's response (an AIMessage) is returned, wrapped in a dictionary to update the messages key in the state. Due to add_messages, this new AI message will be appended to the list.

7. Graph Node Construction: reflect Function (Node):

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

    + This function represents the reflection node.
    + Key Logic: It cleverly inverts the roles of messages. The AI's previous output (AIMessage) is turned into a HumanMessage, and any previous HumanMessage (critiques) are turned into AIMessage. This makes the LLM (when prompted with reflection_prompt) critique what it previously generated as if it were a human submission.
    + The original user request (state["messages"][0]) is kept as is.
    + The reflection_prompt is prepended to these "translated" messages.
    + The LLM's response (the critique) is then wrapped in a HumanMessage. This is crucial because, in the next generate step, this critique will appear as if it's human feedback, guiding the generator to revise its essay.

8. Graph Edge Construction: should_continue Function (Conditional Edge Logic):

    ```python
    def should_continue(state: State):
        if len(state["messages"]) > 8:
            # End after 3 iterations, each with 2 messages
            return END
        else:
            return "reflect"
    ```
    
    + This function determines the next step after the generate node.
    + It checks the total number of messages.
    + The loop starts with 1 user message.
        + Iteration 1: generate (adds 1 AI msg), reflect (adds 1 Human critique msg). Total +2.
        + Iteration 2: generate (adds 1 AI msg), reflect (adds 1 Human critique msg). Total +2.
        + Iteration 3: generate (adds 1 AI msg), reflect (adds 1 Human critique msg). Total +2.
    + If len(state["messages"]) > 8 (meaning 1 initial + 3 iterations * 2 messages/iteration + 1 more generate message = 1+6+1 = 8, so >8 means after the 3rd reflection or 4th generation), the graph transitions to END.
    + Otherwise, it transitions to the "reflect" node.
    + The condition len(state["messages"]) > 8 effectively allows for:
        + Initial User Message
        + Generate 1
        + Reflect 1 (critique of Gen 1)
        + Generate 2 (based on Critique 1)
        + Reflect 2 (critique of Gen 2)
        + Generate 3 (based on Critique 2)
        + Reflect 3 (critique of Gen 3)
        + Generate 4 (based on Critique 3) The graph will end after Generate 4, as len(messages) will be 1 (initial) + 4 (generates) + 3 (reflects) = 8. The next should_continue check after Generate 4 would have len(messages) == 8, so it would go to reflect. After that reflection, len(messages) == 9, and the next generate would be skipped, and the graph would end. 
        
        + Let's re-evaluate:
            + Start: 1 msg
            + After generate: 2 msgs. should_continue -> "reflect"
            + After reflect: 3 msgs.
            + After generate: 4 msgs. should_continue -> "reflect"
            + After reflect: 5 msgs.
            + After generate: 6 msgs. should_continue -> "reflect"
            + After reflect: 7 msgs.
            + After generate: 8 msgs. should_continue -> "reflect"
            + After reflect: 9 msgs. Now len(state["messages"]) > 8 is true. The next time should_continue is called (which would be after the next generate if it happened), it would return END. So, it seems it will do 4 generate-reflect cycles if we count the final generation. The comment "End after 3 iterations, each with 2 messages" implies 1 initial + 3*(generate+reflect) = 1+6 = 7 messages. Then one more generate makes 8. So after the 4th generation, len(messages) is 8. should_continue returns "reflect". After reflection, len(messages) is 9. The next generate node runs, making len(messages) 10. Then should_continue returns END. This means 4 full generate-reflect iterations plus the initial generation. 
            
        + Let's trace:
            + Initial: [H1] (len=1)
            + generate: [H1, A1] (len=2). should_continue -> "reflect"
            + reflect: [H1, A1, H_critique1] (len=3)
            + generate: [H1, A1, H_critique1, A2] (len=4). should_continue -> "reflect"
            + reflect: [H1, A1, H_critique1, A2, H_critique2] (len=5)
            + generate: [H1, A1, H_critique1, A2, H_critique2, A3] (len=6). should_continue -> "reflect"
            + reflect: [H1, ..., A3, H_critique3] (len=7)
            + generate: [H1, ..., H_critique3, A4] (len=8). should_continue -> "reflect"
            + reflect: [H1, ..., A4, H_critique4] (len=9).
            + generate: [H1, ..., H_critique4, A5] (len=10). should_continue -> END. So it performs 5 generations and 4 reflections. The comment might be slightly off or I'm misinterpreting "iteration". If an iteration is (generate + reflect), then 3 iterations means 3 generates and 3 reflects. Initial (1) -> G1 (2) -> R1 (3) -> G2 (4) -> R2 (5) -> G3 (6) -> R3 (7) -> G4 (8). After G4, len(messages) is 8. should_continue returns "reflect". After R4, len(messages) is 9. should_continue (called after the next G5) would return END. So it will run generate 4 times, and reflect 3 times before the condition len(state["messages"]) > 8 stops further reflections and leads to END after the next generation. If the condition was len(state["messages"]) >= 8, it would end sooner. Let's assume "3 iterations" means 3 (generate + reflect) cycles after the first generation. Initial: 1 message. Gen 1: +1 (total 2) Reflect 1: +1 (total 3) Gen 2: +1 (total 4) Reflect 2: +1 (total 5) Gen 3: +1 (total 6) Reflect 3: +1 (total 7) Gen 4: +1 (total 8) At this point, len(state["messages"]) is 8. should_continue will return "reflect". Reflect 4: +1 (total 9). Now, the next node is generate. After generate runs (Gen 5), len(state["messages"]) will be 10. Then should_continue is called: len(state["messages"]) (10) > 8, so it returns END. So, it seems there will be 5 generations and 4 reflections.

9. Graph Construction:

    ```python
    builder = StateGraph(State)
    builder.add_node("generate", generate)
    builder.add_node("reflect", reflect)
    builder.add_edge(START, "generate")
    builder.add_conditional_edges("generate", should_continue)
    builder.add_edge("reflect", "generate")
    graph = builder.compile()
    ```
    A StateGraph is initialized with the defined State.
    generate and reflect functions are added as nodes.
    START is connected to the generate node, meaning the graph execution begins here.
    add_conditional_edges("generate", should_continue): After the generate node runs, the should_continue function is called. Based on its return value ("reflect" or END), the graph transitions to the "reflect" node or finishes.
    builder.add_edge("reflect", "generate"): After the reflect node, the graph always transitions back to the generate node, creating the iterative loop.
    graph.compile(): Compiles the graph definition into a runnable LangGraph object.

10. Graph VisualizationGraph Visualization:

    ```python
        graph_filename = "07-agent-architecture-03/01-reflection-architecture-pattern.png"
        visualize_graph(graph, graph_filename)
    ```
    Calls the visualize_graph utility to save an image of the graph.

11. Graph Execution: Example Usage:

    ```python
    initial_state = {
        "messages": [
            HumanMessage(
                content="Write an essay about the relevance of 'The Little Prince' today."
            )
        ]
    }

    for output in graph.stream(initial_state):
        message_type = "generate" if "generate" in output else "reflect"
        print("\nNew message:", output[message_type]
            ["messages"][-1].content[:], "...")
    ```

    initial_state: The graph starts with a single HumanMessage containing the essay prompt.
    graph.stream(initial_state): Executes the graph. stream yields the output of each node as it executes.
    The loop iterates through the outputs, determines if the output came from the "generate" or "reflect" node, and prints the content of the last message produced by that node. This allows you to see the essay evolving with each generation and the critiques from the reflection step.

12. In summary, this script implements a self-improving essay writer:

    + The user provides an initial essay topic.
    + The generate node writes an initial version of the essay.
    + The reflect node critiques this essay (by making the LLM think it's grading its own previous output).
    + This critique (as a HumanMessage) is fed back to the generate node.
    + The generate node writes a revised essay, taking the critique into account.
    + Steps 3-5 repeat for a few iterations (controlled by should_continue).
    + The process stops, and you have a series of essay versions and critiques.
    + This pattern is powerful for tasks where iterative refinement can lead to higher quality outputs. The key is the "reflection" step, which provides targeted feedback for improvement.

    One minor point of clarity in the should_continue function's comment: The comment "# End after 3 iterations, each with 2 messages" might be slightly ambiguous depending on how "iteration" is defined. As traced above, the condition len(state["messages"]) > 8 leads to 5 generations and 4 reflections. If an "iteration" is defined as one (generate + reflect) cycle, then it completes 4 full such cycles.