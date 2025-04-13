import os
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup # RecursiveUrlLoader uses BeautifulSoup

# --- Optional: Set User-Agent (Good Practice) ---
# Using the direct header approach is often cleaner for loaders
# os.environ["USER_AGENT"] = "MyLangchainLearningBot/1.0 (..." 
# ---

# Define the starting URL
base_url = "https://docs.python.org/3/library/" 
# Note: Be mindful of the site's structure and robots.txt when crawling

try:
    # Initialize the RecursiveUrlLoader
    # - url: The starting point
    # - max_depth: How many levels of links to follow (1 means only the base_url, 2 means base_url + links found on it)
    # - extractor: A function to extract text content from HTML (default uses BeautifulSoup's .get_text())
    # - prevent_outside: Set to True to avoid following links outside the base domain (highly recommended)
    # - use_async: Can potentially speed things up with async requests
    # - headers: Pass custom headers like User-Agent directly
    loader = RecursiveUrlLoader(
        url=base_url, 
        max_depth=2, # Load base_url and pages linked directly from it
        extractor=lambda x: Soup(x, "html.parser").text, # Default extractor
        prevent_outside=True, # Stay within docs.python.org
        use_async=True, # Try async loading
        timeout=600, # Increase timeout for potentially many requests
        # Check documentation for latest way to pass headers if needed, 
        # it might be via a session object or specific kwargs.
        # Example if headers were directly supported:
        # headers={'User-Agent': 'MyLangchainLearningBot/1.0 (...)'} 
    )

    documents = loader.load()

    print(f"\nLoaded {len(documents)} documents recursively.")

    if documents:
        # Print info about a few loaded documents
        for i, doc in enumerate(documents[:3]): # Print first 3
             print(f"\n--- Document {i+1} ---")
             print(f"Source: {doc.metadata.get('source', 'N/A')}")
             print(f"Content Snippet:\n{doc.page_content[:200]}...")
             print("--------------------")
    else:
        print("\nNo documents were loaded. Check the URL and depth.")

except Exception as e:
    print(f"\nAn error occurred during recursive loading: {e}")

