from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from typing import Annotated, TypedDict
from dotenv import load_dotenv
import requests

load_dotenv()

llm = ChatOpenAI()

@tool
def get_stock_price(symbol: str)->dict:
    """Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=9TH1EV1XBL8FTQ8J"
    response = requests.get(url)
    return response.json()

@tool
def purchase_stocks(symbol:str, quantity:int)->dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    if isinstance(decision,str) and decision=="yes":
        return {
            "status":"Success",
            "message":f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol":symbol,
            "quantity":quantity
        }
    else:
        return {
            "status":"Cancelled",
            "message":f"Purchase of {quantity} shares of {symbol} was declined by user.",
            "symbol":symbol,
            "quantity":quantity
        }
    
tools = [get_stock_price, purchase_stocks]
llm_bind_tools = llm.bind_tools(tools)

class chatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state:chatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_bind_tools.invoke(messages)
    return {"messages":[response]}

tool_node = ToolNode(tools)

checkpointer = MemorySaver()

builder = StateGraph(chatState)
builder.add_node("chat",chat_node)
builder.add_node("tools",tool_node)

builder.add_edge(START,"chat")
builder.add_conditional_edges("chat",tools_condition)
builder.add_edge("tools","chat")

chatbot = builder.compile(checkpointer=checkpointer)

if __name__=="__main__":
    thread_id="demo_thread"
    config = {"configurable":{"thread_id":thread_id}}
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit","quit"}:
            break
        state = {"messages":[HumanMessage(content=user_input)]}
        result = chatbot.invoke(state,config=config)
        interrupts = result.get("__interrupt__",[])
        if interrupts:
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").lower().strip()
            result = chatbot.invoke(Command(resume=decision),config=config)
        messages = result["messages"]
        last_message = messages[-1]
        print(f"Bot: {last_message.content}\n")