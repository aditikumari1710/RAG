import os

import pypdf
import json
import warnings
import random


# Define the absolute directory path
directory_path = r"C:\Users\KIIT\Desktop\GithubProj\Research\Data"

# Get a list of all files (including subdirectories) in the specified directory
file_paths = []

for root, dirs, files in os.walk(directory_path):
    for file in files:
        # Join the root directory path with the file name to get the absolute file path
        file_paths.append(os.path.join(root, file))



warnings.filterwarnings("ignore", category=UserWarning, module='PyPDF2')

# Loop through each file
for file_path in file_paths:
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        
        # Initialize a dictionary to hold title and content for each PDF
        doc = {"title": "", "content": {}}
        
        # Extract content from all pages in the PDF
        full_text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            full_text += page.extract_text()
        
        # Split the full text into lines
        lines = full_text.split("\n")
        
        # Extract title (first line of the content)
        title = lines[0] if lines else "No title found"
        
        # Store the title and content directly in the dictionary
        doc["title"] = title
        doc["content"] = {
            "page_content": full_text
        }
        
        # Generate a random number for the output file name
        n = random.randint(1, 10000000)
        
        # Create the output file path
        output_file = f"C:\Users\KIIT\Desktop\GithubProj\Research\Extracted\\output_file_{n}.json"

        # Save the extracted data into the JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=4)
