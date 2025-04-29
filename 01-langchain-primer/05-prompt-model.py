from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
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

# Defining a prompt template
# The PromptTemplate class is used to create a template for the prompt
# The template is a string that contains placeholders for the input variables
# The placeholders are enclosed in curly braces
# The from_template method is used to create a template from a string
# The string can contain any text, including newlines
# The placeholders are replaced with the values of the input variables
# The input variables are passed as a dictionary to the invoke method
# The invoke method is used to generate the prompt
template = PromptTemplate.from_template("""Answer the question based on the context below. 
If the question cannot be answered using the information provided, answer with "I don't know".

Context: {context}

Question: {question}

Answer: """)

# defining the context variable
# The context variable is a string that contains the information that the model can use to answer the question
# The context variable is passed as an input variable to the invoke method
context = """
The most recent advancements in NLP are being driven by Large Language Models (LLMs). 
These models outperform their smaller counterparts and have become invaluable for developers who are creating applications with NLP capabilities. 
Developers can tap into these models through Hugging Face's `transformers` library, 
or by utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.,
"""
# Defining the question variable
# The question variable is a string that contains the question that the model needs to answer
question = "Which model providers offer LLMs?"
# we can use the template to generate the prompt
# The invoke method is used to generate the prompt
# The invoke method takes a dictionary as input, where the keys are the names of the input variables
# and the values are the values of the input variables
# The invoke method returns the generated prompt
# The generated prompt is a string that contains the text of the prompt
generated_prompt = template.invoke(
    {
        "context": context,
        "question": question,
    }
)
# Print the generated prompt
print("\n\nGenerated Prompt:\n")
print(generated_prompt)
print('\n\n')
# create a model instance
# The ChatOpenAI class is used to create a model instance
# The model instance is used to generate the response
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)
# model = ChatOpenAI(model="gpt-3.5o")

# `prompt` and `completion` are the results of using template and model once


response = model.invoke(generated_prompt)
print('\n\nAI Response:\n')
print(response.content)
