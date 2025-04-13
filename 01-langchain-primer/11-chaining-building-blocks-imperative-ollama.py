from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# the building blocks
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

# initialize the model
# the model is a ChatOllama instance
model = ChatOllama(
    model="gemma3:27b",
    temperature=0
)

# combine generation of prompt from prompt template and model invoking in a function
# @chain decorator adds the same Runnable interface for any function you write
@chain
def chatbot(values):
    """A simple chatbot that answers questions."""
    # values is a dictionary with the keys from the template
    # the template will be invoked with the values
    prompt = template.invoke(values)
    return model.invoke(prompt)


# use it
values = {
    "question": "What is the capital of France?"
}
# the function will be invoked with the values
response = chatbot.invoke(values)
print(response.content)
