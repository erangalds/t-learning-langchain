# Enabling Memory for AI Chats with Conversation History
What I learned previously is that, any chat interface API, will not keep a history of your conversation. Which means that, if I say "Hi, I'm eranga, I need your help" and then after a few back and forth conversations, the AI model will not remember what my name is and what we talked about previously. Therefore, to make any conversation with an AI model meaningful and sensible, we need to enable the conversation with memory. 

When I looked at how to do that, what I learned is, its not a hard thing to do. Basically we need to maintain a separate list with all the questions (human / user messages) and answers (AI responses) and if needed the system prompt (main instructions given to control the behavior of the AI model). 

Below steps demonstrates a basic way to incorporate conversation history (a simple form of memory) when interacting with a Large Language Model (LLM) using LangChain, without involving more complex memory management tools like those in LangGraph.

Let's break down the code step-by-step:

1. Imports: 
    As we know, before starting to write the action program logic, we need to *import* all the required *modules* and *libraries*. Since what I'm trying to check is a very simple thing, I just need the `ChatPromptTemplate`, and `ChatOllama` modules. `ChatOllama` is needed because I'm planning to use a local AI model for this test. 

    ```python 
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama import ChatOllama
    ```

    1. `ChatPromptTemplate`: This class is used to create flexible and reusable *prompt structures* for *chat models*. It allows you to define a template with *placeholders* that can be filled in later.
    2. `ChatOllama`: This *class* provides an interface to interact with LLMs hosted by *Ollama* (like gemma3:27b in this case, running locally on my laptop).

2. Defining a Prompt Template:
    Next I'm going to define a *prompt_template* using the `ChatPromptTemplate` module. 

    ```python 
    # Defining a Prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "placeholder",
                "{messages}",
            ),
        ]
    )
    ```

    + `ChatPromptTemplate.from_messages([...])`: This method creates a *prompt template* from a list of *message tuples*. Each tuple typically represents a *role* (like "system", "human", "ai") and its content.
    + `("system", "You are a helpful assistant.")`: This sets a *system message*. System messages are used to give instructions or context to the LLM about its *role* or *behavior*. Here, it's told to be a helpful assistant.
    + `("placeholder", "{messages}")`: This is a crucial part for incorporating memory.
        + "`placeholder`": This special type indicates that the content for this part of the prompt will be a list of messages.
        + "`{messages}`": This is a placeholder variable. When the prompt is formatted (I mean, generated with values substituting the placeholders), a list of actual chat messages (the conversation history) will be substituted here.

3. Initialize the LLM:

    ```python 
    # Initialite the llm
    llm = ChatOllama(
        model="gemma3:27b",
        temperature=0
    )
    ```
    
    + `ChatOllama(...)`: An instance of the `ChatOllama` model is created.
    + `model="gemma3:27b"`: Specifies the Ollama model to be used.
    + `temperature=0`: This parameter controls the randomness of the model's output. A *temperature* of 0 makes the output more *deterministic* and focused, meaning if you give it the same input multiple times, you're more likely to get the same output.

4. Create a Chain:

    Now I have defined the prompt template and the llm. What I need to do now is create a *langchain chain*. This will help me to get the prompt generation from prompte template and calling the llm with the generated prompt by just *invoking* the *chain*. Value of using a *framework*. 

    ```python 
    # Create a chain with the prompt and llm
    chain = prompt | llm
    ```

    + `chain = prompt | llm`: This line uses the LangChain Expression Language (LCEL). The | (pipe) operator chains components together.
    + It means: first, the input will go through the prompt template to be formatted, and then the formatted prompt will be passed to the llm for processing.

5. Defining an Example conversation (Demonstration):

    Now the *chain* is ready. All I need now is to define an example chat conversation with multiple messages belonging to roles like *human*, *ai* etc. Then I can just give that to the *chain* and *invoke* the *chain. Since I'm trying to demonstrate, how to enable memory in a conversation with AI, I delibarately define the conversation in a way to test the AI model to verify whether he can remember what we talked about previously. 

    ```python 
    # Let's Look at the generated prompt
    messages = [
        ("human", "Translate this sentence from English to French: I love programming."),
        ("ai", "J'adore programmer."),
        ("human", "What did you just say?"),
    ]
    
    generated_prompt = prompt.format(messages=messages)
    # Print the generated prompt
    print(f'Generated prompt: \n{generated_prompt}\n')
    ```

    + `messages = [...]`: A list of message tuples is defined. This list represents a conversation history. Each tuple has a role ("human" or "ai") and the content of the message.
    + `generated_prompt = prompt.format(messages=messages)`: The format method of the `ChatPromptTemplate` is used here. It takes a dictionary where keys match the placeholders in the template. This step is unnecessary, because I'm planning to directly *invoke* the *chain*. But I wanted check how the *placeholders* work with *langchain prompt templates*. 
    + The `messages` list is passed to the "`{messages}`" placeholder. LangChain will then format this list into a structure that the chat model understands as a sequence of turns in a conversation.
    + The script then prints this `generated_prompt`. You would see how the system message and the conversation history are combined.

6. Running the Chain:
    Now all I got to do is, *invoking* the *chain* (running the *chain*). 

    ```python 
    # Run the chain with a message
    print("Running the chain with a message...\n")
    response = chain.invoke(
        {
            'messages': messages
        }
    )
    
    # Print the response
    print(response.content)
    ```

    + `chain.invoke({'messages': messages})`: This executes the chain.
    + The input to the chain is a dictionary `{'messages': messages}`.
    + The prompt component in the chain takes this dictionary, uses the messages list to fill its "`{messages}`" placeholder, and generates the full prompt (including the system message and the conversation history).
    + This full prompt is then passed to the llm component.
    + The LLM processes this prompt (which now includes the history "Translate...", "J'adore...", "What did you just say?") and generates a response based on that entire context.
    + `response.content`: The invoke method returns a response object (often an `AIMessage` object). The actual text generated by the LLM is accessed via its `.content` attribute.

In essence, this script shows how to manually construct a conversation history (messages list) and feed it into the LLM via a prompt template. The LLM then uses this history as context to generate its next response.

This is a "simple memory" because the script itself is responsible for maintaining and passing the messages list. For more complex applications, LangChain offers more sophisticated memory modules that can automatically manage and retrieve conversation history.
