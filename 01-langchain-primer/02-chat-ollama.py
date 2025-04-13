from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model="gemma3:27b",
    temperature=0
)
prompt = [HumanMessage("What is the capital of France?")]
# Invoke the model directly
response = llm.invoke(prompt)
# Print the prompt and response
print(f'Prompt:\n{prompt}')
print(f'\nAI Response:\n{response}') 
print(f'\nAI Response Content Only:\n{response.content}')


