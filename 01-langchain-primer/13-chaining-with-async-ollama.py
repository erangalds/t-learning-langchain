from langchain_core.runnables import chain
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# the building blocks
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)
# initialize the model
# the model is a ChatOllama instance
# the model is set to gemma3:27b with a temperature of 0
model = ChatOllama(
    model="gemma3:27b",
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


async def main():
    return await chatbot.ainvoke({"question": "Which model providers offer LLMs?"})

if __name__ == "__main__":
    # asyncio.run is used to run the async function
    # asyncio is a library to write concurrent code using the async/await syntax
    # asyncio.run is used to run the main function
    import asyncio
    print(asyncio.run(main()))
