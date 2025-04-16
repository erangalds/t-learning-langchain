from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    trim_messages,
)
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
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


# Define sample messages
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# Print original messages
print(f'Number of messages: {len(messages)}')
print("Original messages:")
for message in messages:
    print(f"{message}")

# Create trimmer with Ollama Gemma3 model
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=ChatOllama(model="gemma3:27b"),
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Apply trimming
trimmed = trimmer.invoke(messages)
print(f'\nNumber of trimmed messages: {len(trimmed)}')
print("\nTrimmed messages:")
for message in trimmed:
    print(f"{message}")


# Create trimmer with Ollama OpenAI gpt-4o model
print("\n\nUsing OpenAI gpt-4o model for trimming...\n")

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=ChatOpenAI(model="gpt-4o"),
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Apply trimming
trimmed = trimmer.invoke(messages)
print(f'\nNumber of trimmed messages: {len(trimmed)}')
print("\nTrimmed messages:")
for message in trimmed:
    print(f"{message}")
