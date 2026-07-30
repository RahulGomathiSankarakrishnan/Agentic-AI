from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import sqlite3, requests

load_dotenv()

llm = ChatOpenAI()

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator_tool(a:float, b:float, operation:str)->dict:
    """Perform a basic arithmetic operation on two numbers.
    Suported operations - add, sub, mul, div"""

    try:
        if operation == "add":
            result = a+b
        elif operation == "sub":
            result = a-b
        elif operation == "mul":
            result = a*b            
        elif operation == "div":
            if b==0:
                return {"error":"Division by zero is not allowed"}
            result = a/b
        else:
            return {"error":f"Unsupported opertation {operation}"}
        return {"first_num":a,"second_num":b,"operation":operation,"result":result}
    except Exception as e:
        return {"error":str(e)}
    
@tool
def stock_tool(stock:str)->dict:
    """Fetch latest stock for a given symbol (eg 'AAPL', 'TSLA')
    using Alpha vantage with API key in the URL"""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={stock}&apikey=9TH1EV1XBL8FTQ8J"
    response = requests.get(url)
    return response.json()

tools = [search_tool, calculator_tool, stock_tool]

toolnode = ToolNode(tools)

llm_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state:ChatState):
    response = llm_tools.invoke(state['messages'])
    return {'messages':[response]}

graph = StateGraph(ChatState)

graph.add_node("chat_node",chat_node)
graph.add_node("tools",toolnode)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools","chat_node")

conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
cursor = conn.cursor()

checkpointer = SqliteSaver(conn=conn)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = []
    seen = set()
    for checkpoint in checkpointer.list(None):
        t_id = checkpoint.config['configurable']['thread_id']
        if t_id not in seen:
            all_threads.append(t_id)
            seen.add(t_id)
    return all_threads

def create_table():
    cursor.execute("""CREATE TABLE IF NOT EXISTS chat (thread_id TEXT PRIMARY KEY, summary TEXT, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()

def retrieve_all_summaries():
    cursor.execute("""SELECT thread_id, summary FROM chat ORDER BY updated_at DESC""")
    data = cursor.fetchall()
    return [{'thread_id':row[0],'summary':row[1]} for row in data]

def save_summary(thread_id,summary):
    cursor.execute("""
        INSERT INTO chat (thread_id, summary, updated_at) 
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(thread_id) DO UPDATE SET 
            summary=excluded.summary, 
            updated_at=excluded.updated_at
    """, (thread_id, summary))
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