from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# the building blocks
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

# create a model instance
# The ChatOpenAI class is used to create a model instance
# The model instance is used to generate the response
model = ChatOllama(
    model="gemma3:27b", 
    temperature=0
)

# `prompt` and `completion` are the results of using template and model once

# Invoke the model with the generated prompt
# The invoke method is used to generate the response
response = model.invoke(generated_prompt)
# Print the response
print("\n\nAI Response:")
print(response)
print("\n\nAI Response Content Only:")
print(response.content)

