import os
from langchain_community.document_loaders import WebBaseLoader

# --- Set the User-Agent environment variable ---
# Choose a descriptive User-Agent string
# You can mimic a browser or create a custom one for your bot/script
os.environ["USER_AGENT"] = "MyLangchainLearningBot/1.0 (https://mywebsite.com/botinfo; myemail@example.com)" 
# Or a more generic browser one:
# os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
# ---

# Now initialize the loader
loader = WebBaseLoader("https://www.langchain.com/")

try:
    documents = loader.load()
    # Print the number of documents loaded
    print(f"\nLoaded {len(documents)} documents.")
    # Print the first document content
    print(f"\nFirst document content snippet:\n---\n{documents[0].page_content[:200]}...\n---")
except Exception as e:
    print(f"\nAn error occurred during loading: {e}")

