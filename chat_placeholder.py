from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history = [
    HumanMessage(content="Hi"),
    AIMessage(content="Hello! How can I help you?"),
    HumanMessage(content="I ordered a mobile."),
    AIMessage(content="Can you share your order ID?")
]

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "Where is my refund?"
})

response = model.invoke(prompt)

print(response.content)