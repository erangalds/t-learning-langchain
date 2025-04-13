# Import the necessary libraries
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

# the building blocks
model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0
)

prompt = "The sky is"
# Invoke the model directly
response = model.invoke(prompt)
# Print the prompt and response
print(f'\nPrompt:\n{prompt}')
print(f'\n\nAI Response:\n{response}')
print(f'\n\nAI Response Content Only:\n{response.content}')

