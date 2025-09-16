import re
import sys


def extract_paragraphs(tex_file):
    """Extracts paragraphs from a LaTeX file between 'Introduction' and 'Bibliography'.

    Args:
        tex_file (str): Path to the LaTeX file.

    Returns:
        list[str]: A list of strings, where each string is a paragraph.
    """
    with open(tex_file, 'r', encoding='utf-8') as f:  #Handle utf-8 for wider character support
        content = f.read()

    start_pattern = r"\\chapter\{Introduction\}"
    end_pattern = r"\\addcontentsline\{toc\}\{chapter\}\{Bibliography\}"

    start_match = re.search(start_pattern, content)
    end_match = re.search(end_pattern, content)

    if not start_match or not end_match:
        return []  # Return empty list if patterns aren't found

    start_index = start_match.start()
    end_index = end_match.start()

    relevant_content = content[start_index:end_index]

    paragraphs = [p.strip() for p in relevant_content.split('\n\n') if p.strip()] #Split by newline, remove empty strings and whitespace
    return paragraphs



listOfStuff = extract_paragraphs("diss.txt")

for p in listOfStuff:
    print(p)