import os
from langchain_text_splitters import MarkdownTextSplitter
# If loading from file into Document objects first:
# from langchain_community.document_loaders import TextLoader

# --- Configuration ---
# Assuming this script is in a directory one level below the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_folder = os.path.join(project_root, "sample-data")
md_file_name = "gemini-code-assist.md" # Your Markdown file
md_file_path = os.path.join(data_folder, md_file_name)

print(f"Attempting to read Markdown file from: {md_file_path}")

# Check if the file exists
if not os.path.isfile(md_file_path):
    print(f"Error: Markdown file not found at {md_file_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()
# --- End Configuration ---

try:
    # Read the content of the Markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    print(f"\nSuccessfully read {len(markdown_content)} characters from the file.")

    # Initialize the MarkdownTextSplitter
    # It understands Markdown structure inherently.
    markdown_splitter = MarkdownTextSplitter(
        chunk_size=250,  # Adjust chunk size as needed
        chunk_overlap=25   # Adjust overlap as needed
    )

    # Split the Markdown content
    # split_text works directly on the string content
    chunks = markdown_splitter.split_text(markdown_content)

    print(f"\nSplit the Markdown into {len(chunks)} chunks using MarkdownTextSplitter.")

    if chunks:
        print("\n--- First 3 Chunks ---")
        for i, chunk in enumerate(chunks[:3]):
            # Note: The chunks here are strings. If you used a loader first
            # and split_documents, they would be Document objects.
            print(f"\n[Chunk {i+1}] (Length: {len(chunk)} characters)")
            print(f"Content:\n'''\n{chunk}\n'''") # Use triple quotes for clarity
            print("-" * 20)

    # If you loaded with TextLoader first:
    # loader = TextLoader(md_file_path, encoding='utf-8')
    # documents = loader.load()
    # chunks_docs = markdown_splitter.split_documents(documents)
    # print(f"\nSplit into {len(chunks_docs)} Document objects.")
    # if chunks_docs:
    #     print(f"\nContent of first chunk doc:\n'''\n{chunks_docs[0].page_content}\n'''")
    #     print(f"Metadata: {chunks_docs[0].metadata}")


except FileNotFoundError:
    print(f"Error: File not found at {md_file_path}")
except Exception as e:
    print(f"\nAn error occurred: {e}")

