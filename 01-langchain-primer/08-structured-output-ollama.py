from langchain_ollama import ChatOllama
from pydantic import BaseModel
import os 
from dotenv import load_dotenv # Import load_dotenv
load_dotenv() # Load environment variables from .env file

# Set the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
# Check if the API key is set
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set")

class AnswerWithJustification(BaseModel):
    """An answer to the user's question along with justification for the answer."""

    answer: str
    """The answer to the user's question"""
    justification: str
    """Justification for the answer"""

# the building blocks
# initialize the model
llm = ChatOllama(
    model="gemma3:27b", 
    temperature=0
)
# Extracting the answer and justification from the model's response
# The model will return a structured response
# `with_structured_output` is a method that allows you to specify the output format
# for the model's response. In this case, it is set to return an instance of
# `AnswerWithJustification`, which is a Pydantic model.
# This means that the model's response will be parsed and validated against the
# `AnswerWithJustification` model, and the attributes of the model will be
# accessible as properties of the response object.
structured_llm = llm.with_structured_output(AnswerWithJustification)

prompt = "What weighs more, a pound of bricks or a pound of feathers"

response = structured_llm.invoke(prompt)
# Extract the answer and justification from the response
answer = response.answer
# Extract the justification from the response
justification = response.justification

print(f'\nQuestion: {prompt}')
print(f'\n\nAnswer: {answer}')
print(f'\n\nJustification: {justification}')


