from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
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

# Since we are using the ChatOpenAI function, we need to use the HumanMessage class
# to create the prompt
# Create a prompt using the HumanMessage class
# This is a single message prompt
# In this case, we are asking the model a question
# The HumanMessage class is used to create a message from the user
prompt = [HumanMessage("What is the capital of France?")]

# Invoke the model directly
response = model.invoke(prompt)
# Print the prompt and response
print(f'Prompt:\n{prompt}')
print(f'\n\nAI Response:\n{response}')
print(f'\n\nAI Response Content Only:\n{response.content}')

