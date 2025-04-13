from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gemma3:27b",
    temperature=0
)
# `invoke` is a method that allows you to send a prompt to the model
completion = model.invoke("Hi there!")
print(f'\nReponse with invoke:\n{completion.content}')

# `batch` is a method that allows you to send multiple prompts to the model 
# and get multiple responses
# in a single call
# This is more efficient than calling `invoke` multiple times
completions = model.batch(["Hi there!", "Bye!"])    
print(f'\n\nResponse with batch:\n{completion.content}')


# `stream` is a method that allows you to get the response from the model
# token by token
# This is useful for getting a streaming response
# from the model
print(f'\n\nResponse with stream:\n')
for token in model.stream("Bye!"):
    # Print the token content
    # This is useful for getting a streaming response
    print(token.content)    

