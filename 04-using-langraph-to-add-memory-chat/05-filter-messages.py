from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
)

# Sample messages
messages = [
    SystemMessage(content="you are a good assistant", id="1"),
    HumanMessage(content="example input", id="2", name="example_user"),
    AIMessage(content="example output", id="3", name="example_assistant"),
    HumanMessage(content="real input", id="4", name="bob"),
    AIMessage(content="real output", id="5", name="alice"),
]

# Print original messages
print(f'Number of messages: {len(messages)}')
print("Original messages:")
for message in messages:
    print(message)
# Filter for human messages
human_messages = filter_messages(messages, include_types="human")
print("\n\nHuman messages:")
for message in human_messages:
    print(message)
    

# Filter to exclude certain names
excluded_names = filter_messages(
    messages, exclude_names=["example_user", "example_assistant"]
)
print("\n\nExcluding example names:")
for message in excluded_names:
    print(message)

# Filter by types and IDs
filtered_messages = filter_messages(
    messages, include_types=["human", "ai"], exclude_ids=["3"]
)
print("\n\nFiltered by types and IDs:")
for message in filtered_messages:
    print(message)
