from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOllama(
    model="gemma3:27b",
    temperature=0
)
# Defining a system message
# The system message is used to set the behavior of the assistant
# In this case, we are setting the assistant to be helpful
# and respond to questions with three exclamation marks
system_msg = SystemMessage(
    "You are a helpful assistant that responds to questions with three exclamation marks."
)
# Defining a human message
# The human message is the input from the user
# In this case, we are asking the model a question
# The HumanMessage class is used to create a message from the user
human_msg = HumanMessage("What is the capital of France?")
# Invoke the model directly
response = model.invoke([system_msg, human_msg])
# Print the prompt and response
print(f'Prompt:\n{[system_msg, human_msg]}')
print(f'\n\nAI Response:\n{response}')
print(f'\n\nAI Response Content Only:\n{response.content}')
