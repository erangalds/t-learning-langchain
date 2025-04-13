from langchain_core.runnables import chain
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
# The model is set to gpt-4o-mini with a temperature of 0
model = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0
)
# combine them in a function
# @chain decorator adds the same Runnable interface for any function you write
# async keyword makes it an async function
@chain
async def chatbot(values):
    prompt = await template.ainvoke(values)
    # await keyword make it to wait till model.ainvoke is done
    return await model.ainvoke(prompt)

# Defining a main function to run the chatbot. But this main function will be async
async def main():
    return await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})

if __name__ == "__main__":
    # asyncio.run is used to run the async function
    # asyncio is a library to write concurrent code using the async/await syntax
    # asyncio.run is used to run the main function
    import asyncio
    print(asyncio.run(main()))
