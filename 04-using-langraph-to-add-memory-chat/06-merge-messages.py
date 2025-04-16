from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)

# Sample messages with consecutive messages of same type
messages = [
    SystemMessage(content="you're a good assistant."),
    SystemMessage(content="you always respond with a joke."),
    HumanMessage(
        content=[{"type": "text", "text": "i wonder why it's called langchain"}]
    ),
    HumanMessage(content="and who is harrison chasing anyways"),
    AIMessage(
        content='Well, I guess they thought "WordRope" and "SentenceString" just didn\'t have the same ring to it!'
    ),
    AIMessage(
        content="Why, he's probably chasing after the last cup of coffee in the office!"
    ),
]

# Print original messages
print(f'Number of messages: {len(messages)}')
print("Original messages:")
for message in messages:
    print(f"{message}")
# Merge consecutive messages
merged = merge_message_runs(messages)
print(f'\nNumber of merged messages: {len(merged)}')
print("\nMerged messages:")
for message in merged:
    print(f"{message}")
