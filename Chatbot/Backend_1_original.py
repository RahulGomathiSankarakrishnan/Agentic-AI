from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state:ChatState):
    response = llm.invoke(state['messages'])
    return {'messages':[response]}

graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

# thread_id='2'
# while True:
#     user_message = input("Type here: ")
#     print("User: ",user_message)
#     if user_message.strip().lower() in ["quit","bye","exit"]:
#         break
#     config = {'configurable':{'thread_id':thread_id}}
#     response = chatbot.invoke({'messages':HumanMessage(content=user_message)},config=config)
#     print('AI: ',response['messages'][-1].content)

# print(list(chatbot.get_state_history(config={'configurable':{'thread_id':thread_id}})))