from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

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

# Initialite the llm 
llm = ChatOllama(
    model="gemma3:27b", 
    temperature=0
)

# Create a chain with the prompt and llm
chain = prompt | llm 

# Let's Look at the generated prompt
messages = [
    ("human", "Translate this sentence from English to French: I love programming."),
    ("ai", "J'adore programmer."),
    ("human", "What did you just say?"),
]

generated_prompt = prompt.format(messages=messages)
# Print the generated prompt
print(f'Generated prompt: \n{generated_prompt}\n')

# Run the chain with a message
print("Running the chain with a message...\n")
response = chain.invoke(
    {
        'messages': messages
    }
)

# Print the response
print(response.content)