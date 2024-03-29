#print("hello world")
from openai import OpenAI
import pyttsx3

client = OpenAI()

TTS_engine = pyttsx3.init()

voices = TTS_engine.getProperty("voices")
TTS_engine.setProperty("voice", voices[1].id)
print(voices)

def chatgpt_message_handler(m_mess = [], c_mess = []):

    if m_mess == []:
        movement_messages = [
            {
                "role": "system", 
                "content": "You are a text to movement description assistant, you control a 3d humanoid figure, and your task is to generate a one sentence description of the appropriate movements that the figure should take in response to the user."
            },
            {
                "role": "user", 
                "content": "The user is trying to learn a simple dance, in which they must raise their left arm, then turn in a circle, then jump twice."
            }
        ]
    else:
        movement_messages = m_mess
    if c_mess == []:
        conversation_messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant, and should always respond with no more than a single sentence."
            },
            {
                "role": "user", 
                "content": "The user is trying to learn a simple dance, in which they must raise their left arm, then turn in a circle, then jump twice."
            }
        ]
    else:
        conversation_messages = c_mess
    movement_conversation = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = movement_messages
    )
    standard_conversation = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages = conversation_messages
    )
    print(movement_conversation.choices[0].message.content)
    print(standard_conversation.choices[0].message.content)

    input_filename = "input.txt"
    b = open(input_filename, "w", encoding="iso-8859-1")
    b.write(movement_conversation.choices[0].message.content)
    b.close()


    movement_messages.append(
        {
            "role":"assistant",
            "content":movement_conversation.choices[0].message.content
        }
    )
    conversation_messages.append(
        {
            "role":"assistant",
            "content":standard_conversation.choices[0].message.content
        }
    )

    TTS_engine.say(standard_conversation.choices[0].message.content)
    # play the speech
    TTS_engine.runAndWait()

    continue_ = input("Please enter the desired response to the ai, or type exit to close the program: ")

    if continue_ != "exit":
        movement_messages.append(
            {
                "role":"user",
                "content":continue_
            }
        )
        conversation_messages.append(
            {
                "role":"user",
                "content":continue_
            }
        )
        chatgpt_message_handler(movement_messages, conversation_messages)


chatgpt_message_handler()

"""def write_list_and_export_to_UE5_project(list):
    #filepath = "//..//MyProject3//Animations//"
    filepath = "../../MyProject3/Animations/"
    filename = filepath + "Anim.txt"
    a = open(filename, "w", encoding="iso-8859-1")
    a.write("--- File displays animations as a large set of lists ---\n")
    #print("starting loop")
    for frame in list:
        a.write("Frame: \n")
        for point in frame:
            a.write("Point: \n[\n")
            for coordinate in point:
                a.write("\t" + str(coordinate) + "\n")
            a.write("]\n")
    a.close()
    #print("finished loop")

print("starting")

new_list = [[[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2], [0, 1, 2]]]

#write_list_and_export_to_UE5_project(new_list)

print("successful?")"""