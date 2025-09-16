import sys
import time
import lmstudio as lms
from lmstudio import Chat
import datetime
if __name__ == "__main__":
    now = datetime.datetime.now()
    print(now)
    num1 = sys.argv[1]

    whichNumber = int(num1)
    # "qwq-32b@q8_0"
    # "gemma-3-27b-it@q4_k_m"
    model = lms.llm("gemma-3-4b-it", config={
        "contextLength": 65536,
     "temperature": 0.1,
      "maxTokens": False,
    })

    with open('systemPrompt.txt', 'r', encoding='utf-8') as file:
      systemPrompt = file.read()
    chat = Chat(systemPrompt)
    # chat = Chat("You are a resident AI philosopher.")
    with open('diss.txt', 'r', encoding='utf-8') as file:
      diss = file.read()

    with open('between.txt', 'r', encoding='utf-8') as file:
      between = file.read()

    with open('reformulateTextList/reformulateText'+str(whichNumber)+'.txt', 'r', encoding='utf-8') as file:
      reformulateText = file.read()

    inputText = diss + between+ reformulateText + "\n remember this is your task: \n" + systemPrompt
    chat.add_user_message(inputText)

    prediction = model.respond(chat)
    print(prediction)
    with open('reformulateTextOutput/Output'+str(whichNumber)+'.txt', "w") as text_file:
        text_file.write(prediction.content)

    now = datetime.datetime.now()
    print(now)