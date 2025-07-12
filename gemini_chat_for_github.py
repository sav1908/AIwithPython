
import google.generativeai as genai

#use your own api key which you can get from aistudio.google.com, go to create api key and paste in the line below
API_KEY = 'your own api key'
genai.configure(api_key = API_KEY)


model = genai.GenerativeModel("gemini-2.0-flash")
chat = model.start_chat()

print("chat with gemini, type 'exit' to quit")
while True:
    uin = input("you: ")
    if uin.lower() == 'exit':
        break

    response = chat.send_message(uin)
    print('gemini: ', response.text)