from langchain_core.runnables import chain
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# initialize the model
# the model is a ChatOllama instance
model = ChatOllama(
    model="gemma3:27b",
    temperature=0
)

# the building blocks
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

# `@chain` decorator adds the same Runnable interface for any function you write
# and makes it streamable
# this is a generator function
@chain
def chatbot(values):
    """A simple chatbot that answers questions."""
    # values is a dictionary with the keys from the template
    # the template will be invoked with the values
    # and the model will be interfaced with stream interface
    prompt = template.invoke(values)
    for token in model.stream(prompt):
        yield token.content

# use it
for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
    print(part)
