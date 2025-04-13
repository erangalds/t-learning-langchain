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

# `ChatPromptTemplate` is a class that allows you to create a prompt template for chat models.
# It is a subclass of `PromptTemplate` and is used to create prompts for chat models like GPT-3.5 and GPT-4.
# It allows you to define a template for the prompt and fill in the variables at runtime.
# The `from_messages` method is a class method that creates a `ChatPromptTemplate` from a list of messages.
# Each message is a tuple of the form (role, content), where role is either "system", "user", or "assistant".
# The `system` message is used to set the behavior of the assistant.
# The `user` message is used to set the input from the user.
# The `assistant` message is used to set the output from the assistant.
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            'Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don\'t know".',
        ),
        ("human", "Context: {context}"),
        ("human", "Question: {question}"),
    ]
)
# defining the context varialbe
# The context variable is a string that contains the information that the model can use to answer the question
context = """
The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. 
Developers can tap into these models through Hugging Face's `transformers` library, 
or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.
"""
# Defining the question variable
# The question variable is a string that contains the question that the model needs to answer
question = "Which model providers offer LLMs?"
# we can use the template to generate the prompt
# The invoke method is used to generate the prompt
# The invoke method takes a dictionary as input, where the keys are the names of the input variables
generated_prompt = template.invoke(
    {
        "context": context,
        "question": question,
    }
)
# Print the generated prompt
# The generated prompt is a string that contains the text of the prompt
print("Generated Prompt:")
print(generated_prompt)
# create a model instance
# The ChatOpenAI class is used to create a model instance
# The model instance is used to generate the response
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# `prompt` and `completion` are the results of using template and model once
response = model.invoke(generated_prompt)
content = response.content

# Print the response
print("\n\nAI Response:")
print(response)
print("\n\nAI Response Content Only:")  
print(content)
