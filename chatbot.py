from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]

while True:
    user_input = input("You: ")

    if user_input == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    result = model.invoke(chat_history)

    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

print(chat_history)