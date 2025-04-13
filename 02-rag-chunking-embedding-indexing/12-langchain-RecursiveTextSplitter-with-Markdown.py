import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# If loading from file:
# from langchain_community.document_loaders import TextLoader

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_folder = os.path.join(project_root, "sample-data")
md_file_name = "gemini-code-assist.md" # Ensure this file exists
md_file_path = os.path.join(data_folder, md_file_name)

print(f"Attempting to read Markdown file from: {md_file_path}")

if not os.path.isfile(md_file_path):
    print(f"Error: Markdown file not found at {md_file_path}")
    exit()
# --- End Configuration ---

try:
    # Read the content of the Markdown file
    with open(md_file_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    print(f"\nSuccessfully read {len(markdown_content)} characters from the file.")

    # Initialize the splitter using the from_language class method
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, # Specify the language
        chunk_size=250,             # Adjust chunk size as needed
        chunk_overlap=25             # Adjust overlap as needed
        # No need to provide 'separators' manually here
    )

    # Split the Markdown content
    chunks = markdown_splitter.split_text(markdown_content)

    print(f"\nSplit the Markdown into {len(chunks)} chunks using language-specific separators.")

    if chunks:
        print("\n--- First 3 Chunks ---")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[Chunk {i+1}] (Length: {len(chunk)} characters)")
            print(f"Content:\n'''\n{chunk}\n'''")
            print("-" * 20)

except FileNotFoundError:
    print(f"Error: File not found at {md_file_path}")
except Exception as e:
    print(f"\nAn error occurred: {e}")

