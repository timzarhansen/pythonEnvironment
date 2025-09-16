import sys
import time
import lmstudio as lms
from lmstudio import Chat
import datetime

import re


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


def remove_think_sections(text: str) -> str:
    """
    Removes all sections enclosed by <think> and </think> (including the tags).

    Args:
        text (str): The input string

    Returns:
        str: The string with <think>...</think> sections removed
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def format_number(n: int) -> str:
    return f"{n:04d}"


if __name__ == "__main__":
    now = datetime.datetime.now()
    print(now)


    whichNumber = 1
    # "qwq-32b@q8_0"
    # "gemma-3-27b-it@q4_k_m"
    # "gemma-3-4b-it"
    model = lms.llm("qwq-32b@q8_0", config={
        "contextLength": 65536,
     "temperature": 0.6,
      "maxTokens": False,
    })

    # model = lms.llm("gemma-3-4b-it", config={
    #     "contextLength": 65536,
    #  "temperature": 0.1,
    #   "maxTokens": False,
    # })

    with open('systemPrompt.txt', 'r', encoding='utf-8') as file:
      systemPrompt = file.read()

    # chat = Chat("You are a resident AI philosopher.")
    with open('diss.txt', 'r', encoding='utf-8') as file:
      diss = file.read()

    with open('between.txt', 'r', encoding='utf-8') as file:
      between = file.read()

    listOfStuff = extract_paragraphs("diss.txt")
    for stuff in listOfStuff:
        if whichNumber>120:
            inputText = diss + between+ stuff + "\n remember this is your task: \n" + systemPrompt
            chat = Chat(systemPrompt)
            chat.add_user_message(inputText)

            prediction = model.respond(chat)
            # print(prediction)
            with open('reformulateTextOutput/'+format_number(whichNumber)+'_Output.txt', "w") as text_file:
                text_file.write(remove_think_sections(prediction.content))
            with open('reformulateTextOutput/'+format_number(whichNumber)+'_Input.txt', "w") as text_file:
                text_file.write(stuff)
            now = datetime.datetime.now()
            print(now)
            print(whichNumber)
            # if whichNumber>45:
            #     break
        whichNumber = whichNumber + 1