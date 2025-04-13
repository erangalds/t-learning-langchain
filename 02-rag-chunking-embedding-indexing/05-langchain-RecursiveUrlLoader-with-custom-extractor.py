import os
import re # Import regular expressions module
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

# --- Optional: Set User-Agent ---
# os.environ["USER_AGENT"] = "MyLangchainLearningBot/1.0 (...)"
# ---

# Custom extractor function
def simple_extractor(html: str) -> str:
    soup = Soup(html, "html.parser")
    # Get text, strip leading/trailing whitespace from each line,
    # remove blank lines, and join with a single newline.
    text_lines = [line.strip() for line in soup.get_text().splitlines()]
    # Filter out empty lines after stripping
    non_empty_lines = [line for line in text_lines if line]
    # Join the non-empty lines with a single newline
    cleaned_text = "\n".join(non_empty_lines)
    # Optional: Replace multiple spaces/tabs within lines with a single space
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)
    return cleaned_text

# Define the starting URL
base_url = "https://www.langchain.com/" # Example URL

try:
    loader = RecursiveUrlLoader(
        url=base_url,
        max_depth=1, # Only the base page for this example
        extractor=simple_extractor, # Use our custom function
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # headers={'User-Agent': 'MyLangchainLearningBot/1.0 (...)'}
    )

    documents = loader.load()
    print(f"\nLoaded {len(documents)} documents with custom extractor.")
    if documents:
        print(f"\nCleaned content snippet:\n---\n{documents[0].page_content[:300]}...\n---")

except Exception as e:
    print(f"\nAn error occurred during recursive loading: {e}")

