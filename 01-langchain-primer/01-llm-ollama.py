from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(
    model="gemma3:27b", 
    temperature=0
)

prompt = "The sky is"
# Invoke the model directly
response = model.invoke(prompt)
# Print the prompt and response
print(f'Prompt:\n{prompt}')
print(f'\nAI Response:\n{response}')

