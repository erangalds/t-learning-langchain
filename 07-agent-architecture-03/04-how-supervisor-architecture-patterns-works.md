# Building a AI workflow with Supervisor Agent Architecture Pattern

## Overall Purpose:

The script sets up a multi-agent system where a "supervisor" agent decides which specialized agent ("researcher" or "coder") should handle the current step of a task, based on a user's request and the conversation history. The goal is to complete the user's request by routing it to the appropriate worker agent until the task is finished.

## Key Components:

1.  Authentication (`authenticate_with_openai`):

    + This function is designed to load an OpenAI API key from a `.env` file.
    + It's called at the beginning of the script. However, the `ChatOpenAI` model, which would use this key, is commented out in favor of `ChatOllama`. So, in the current active configuration, this function's primary purpose (setting the OpenAI key for an OpenAI model) isn't directly utilized by the active LLM.

2. State Definitions:

    + `SupervisorDecision(BaseModel)`: This Pydantic model defines the expected output structure for the supervisor agent. It ensures the supervisor's decision will be one of "researcher", "coder", or "FINISH".
    + `AgentState(MessagesState)`: This inherits from `MessagesState` (which is designed to hold a list of messages) and adds a `next` field. This `next` field will store the supervisor's decision on which agent should act next or if the process should finish.
        + Inheriting from `MessageState` class in *LangGraph* provides a convenient and standardized way to manage the conversation history within your graph's state
            + **Benefit**:
                + Automatic Message Handling: *LangGraph* knows how to automatically handle updates to the `messages` list when a node in your graph returns a `BaseMessage`  object like `HumanMessage`, `AIMessage`, `SystemMessage`, etc. or a list of `BaseMessage` objects. 
            + **Inherited Attributes**:
                + `messages` attribute: Which is a list (`List[BaseMessgae`]). This list holds the sequence of messages that represent the conversation history.

3. Model Initialization:

    + `model = ChatOllama(...)`: Initializes a local LLM (Gemma 3 27B) using `ChatOllama`. This is the primary model used by the supervisor and the worker agents in the active configuration.
    + `supervisor_model = model.with_structured_output(SupervisorDecision)`: This is crucial. It takes the model and wraps it so that its output is forced to conform to the `SupervisorDecision` Pydantic model. This means the supervisor LLM will always output a decision in the specified format.
    + `llm = ChatOllama(...)`: Another instance of ChatOllama, used by the "researcher" and "coder" agents to generate their responses.
    + Commented out `ChatOpenAI` lines show an alternative setup using OpenAI's GPT models.

4. Agent Definitions:

    + `agents = ["researcher", "coder"]`: A list of available worker agents.

5. System Prompts:

    + `system_prompt_part_1`: Instructs the supervisor LLM on its role: to manage the conversation between workers and decide the next worker based on the user request.
    + `system_prompt_part_2`: Further instructs the supervisor to check if the user's request has been satisfied by the AI's response and to output "FINISH" if so, otherwise select the next worker.

6. Node Functions (Agent Logic):

    + `supervisor(state)`:
        + This is the core of the supervisor agent.
        + It constructs a list of messages, including the system prompts and the current conversation history (`state["messages"]`).
        + It invokes `supervisor_model` with these `messages`. Because `supervisor_model` is configured for structured output, the result (`decision_obj`) will be an instance of `SupervisorDecision`.
        + It returns a dictionary updating the next field in the state with the supervisor's decision (e.g., "researcher", "coder", or "FINISH").
    + `researcher(state: AgentState)`:
        + Represents the "researcher" agent.
        + It takes the first message from the state (`state["messages"][0].content`) as the user's query.
        + It invokes the `llm` with a system prompt telling it to act as a research assistant and the user's query.
        + It returns the LLM's response as a new message to be added to the state.
    + `coder(state: AgentState)`:
        + Represents the "coder" agent.
        + Similar to the researcher, it takes the first message's content.
        + It invokes the llm with a system prompt telling it to act as a coding assistant.
        + It returns the LLM's response as a new message.

7. Graph Construction (`StateGraph`):

    + `builder = StateGraph(AgentState)`: Initializes a state graph that will manage the `AgentState`.
    + `builder.add_node("supervisor", supervisor)`: Adds the `supervisor` function as a node named "supervisor".
    + `builder.add_node("researcher", researcher)`: Adds the `researcher` function as a node.
    + `builder.add_node("coder", coder)`: Adds the `coder` function as a node.
    + `builder.add_edge(START, "supervisor")`: Defines the entry point of the graph. When the graph starts, it will first execute the "supervisor" node.
    + `builder.add_conditional_edges("supervisor", lambda state: state["next"])`: This is a key part of the `supervisor` pattern. After the "supervisor" node runs, this edge directs the flow to another node based on the value of state["next"].
        + If `state["next"]` is "researcher", it goes to the "researcher" node.
        + If `state["next"]` is "coder", it goes to the "coder" node.
        + If `state["next"]` is "FINISH", the graph execution for this branch effectively ends (though "FINISH" itself isn't a node here, the conditional routing stops if no edge matches "FINISH"). LangGraph typically uses END for explicit termination.
    + `builder.add_edge("researcher", "supervisor")`: After the "researcher" node runs, control returns to the "supervisor" node.
    + `builder.add_edge("coder", "supervisor")`: Similarly, after the "coder" node runs, control returns to the "supervisor".
    + `graph = builder.compile()`: Compiles the defined graph into an executable object.

8. Example Usage:

    + `initial_state`: Sets up the initial state for the graph run. It includes a user message asking for help with a Python task involving Excel and CSV files. It also initializes `next` to "supervisor", though the `START` edge already directs to the supervisor.
    + for output in `graph.stream(initial_state): ...`: This runs the graph with the `initial_state`.
        + `graph.stream()` executes the graph step-by-step, yielding the output of each node as it runs.
        + The loop prints the output from each step, allowing you to see the flow of control and the messages generated.

How it Works (Flow):

1. The graph starts, and the `supervisor` node is called.
2. The `supervisor` LLM analyzes the user's request ("I need help to open an excel file...") and the conversation history (initially just the user's request).
3. Based on its prompts, the `supervisor_model` decides which agent is best suited next (e.g., "coder" for a coding task, or "researcher" if more information is needed). Let's assume it picks "coder". It returns {"next": "coder"}.
4. The conditional edge from "supervisor" routes the execution to the "coder" node.
5. The `coder`node is called. It takes the user's request, invokes its LLM (instructed to be a coding assistant), and generates a Python code snippet or explanation. It returns `{"messages": [coder_response_message]}`.
6. The edge from "coder" routes execution back to the "supervisor" node.
7. The `supervisor` node is called again. Now, `state["messages"]` includes the original user request and the coder's response.
8. The `supervisor` LLM re-evaluates. If the coder's response satisfies the user's request, it might decide "FINISH". If the request needs further refinement or another step (e.g., research on a specific library), it might choose "researcher" or "coder" again.
9. This loop continues until the `supervisor` decides "FINISH" or a predefined limit/condition is met.

In essence, this script creates an autonomous loop where a manager (supervisor) delegates tasks to workers (coder, researcher) and reviews their output, deciding the next course of action until the overall goal is achieved. The use of `with_structured_output` for the supervisor is a good practice to ensure its decisions are machine-readable and can be used for routing.

