from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import sqlite3

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

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
cursor = conn.cursor()


checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def create_table():
    cursor.execute("""CREATE TABLE IF NOT EXISTS chat_summaries (thread_id TEXT PRIMARY KEY, summary TEXT)""")
    conn.commit()

def retrieve_all_summaries():
    cursor.execute("""SELECT thread_id, summary FROM chat_summaries""")
    data = cursor.fetchall()
    return [{'thread_id':row[0],'summary':row[1]} for row in data]

def save_summary(thread_id,summary):
    cursor.execute("""INSERT OR REPLACE INTO chat_summaries VALUES (?,?)""", (thread_id,summary))
    conn.commit()


# thread_id='2'
# while True:
    # user_message = input("Type here: ")
#     print("User: ",user_message)
#     if user_message.strip().lower() in ["quit","bye","exit"]:
#         break
    # config = {'configurable':{'thread_id':thread_id}}
    # response = chatbot.invoke({'messages':HumanMessage(content=user_message)},config=config)
    # print('AI: ',response['messages'][-1].content)

# print(list(chatbot.get_state_history(config={'configurable':{'thread_id':thread_id}})))