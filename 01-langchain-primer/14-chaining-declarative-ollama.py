from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# the building blocks
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

model = ChatOllama(
    model="gemma3:27b",
    temperature=0
)

# chaining / combine them with the | operator
# The | operator is used to combine the template and model
# The template will be invoked with the values and
# then the model will be invoked with the prompt
# The response will be a ChatMessage object
chatbot = template | model

# use it
response = chatbot.invoke({"question": "Which model providers offer LLMs?"})

print(response.content)

# The same declarative code can be used for streaming 
# We don't need to change the code for streaming
# streaming
for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
    print(part)
