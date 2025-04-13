from langchain_openai.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv # Import load_dotenv

# defining the function to authenticate with OpenAI
def authenticate_with_openai():
    """
    Authenticate with OpenAI using the API key from environment variables.
    """
    dotenv_path = "../.env" # Path to the .env file
    # Load environment variables from the .env file
    load_dotenv(dotenv_path=dotenv_path) 
    # Load the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Check if the API key is set
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set")

# Authenticate with OpenAI
authenticate_with_openai()# This function sets the OpenAI API key from an environment variable

# Initialize the model
model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0
)

# `invoke` is a method that allows you to send a prompt to the model
completion = model.invoke("Hi there!")
print(f'Response with invoke:\n{completion}')

# `batch` is a method that allows you to send multiple prompts to the model 
# and get multiple responses
# in a single call
# This is more efficient than calling `invoke` multiple times
completions = model.batch(["Hi there!", "Bye!"])    
print(f'\n\nResponse with batch:\n{completions}')

# `stream` is a method that allows you to get the response from the model
# token by token
# This is useful for getting a streaming response
# from the model
print(f'\n\nResponse with stream:\n')
for token in model.stream("Bye!"):
    # Print the token content
    # This is useful for getting a streaming response
    print(token.content)    

