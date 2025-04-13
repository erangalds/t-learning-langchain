from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
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

# Defining the prompt template
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)

# initialize the model
# Using the ChatOpenAI class to create a model instance
# The model is set to gpt-4o-mini with a temperature of 0
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
# This will invoke the template with the values and 
# then invoke the model with the prompt
# The response will be a ChatMessage object
# The content of the response will be the model's response
response = chatbot.invoke(values)
print(response.content)
