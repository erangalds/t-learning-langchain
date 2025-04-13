from langchain_core.output_parsers import CommaSeparatedListOutputParser

# `CommaSeparatedListOutputParser` is used to parse a comma-separated list of items
# It is a subclass of `OutputParser` and is used to parse the output of a model into a list of items
# The `invoke` method is used to parse the output of the model
# The `invoke` method takes a string as input and returns a list of items
parser = CommaSeparatedListOutputParser()

response = parser.invoke("apple, banana, cherry")
print(response)
