from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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

# Initialize the model
model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0
)

# chaining / combine them with the | operator
# The | operator is used to combine the template and model
# The template will be invoked with the values and
# then the model will be invoked with the prompt
# The response will be a ChatMessage object
chatbot = template | model

# use it
response = chatbot.invoke({"question": "Which model providers offer LLMs?"})

print(response.content)

# The same declarative code can be used for streaming 
# We don't need to change the code for streaming
# streaming
for part in chatbot.stream({"question": "Which model providers offer LLMs?"}):
    print(part)
