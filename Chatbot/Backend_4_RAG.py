from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from typing import TypedDict, Annotated, Dict, Any, Optional
from dotenv import load_dotenv
import sqlite3, requests, os, tempfile

load_dotenv()

llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()

# --- 2. PER-THREAD RAG STORAGE ---
# These act as in-memory lookups to keep PDF contexts isolated by thread_id
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Builds a FAISS retriever for the uploaded PDF and stores it for the thread.
    
    Returns a summary dict that can be surfaced in the UI."""
    
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
        chunks = splitter.split_documents(docs)

        # Create an in-memory FAISS store for this specific thread
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or "Uploaded PDF",
            "documents":len(docs),
            "chunks": len(chunks)
        }
        return _THREAD_METADATA[str(thread_id)]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


#Tools
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

@tool
def rag_tool(query: str, thread_id: Optional[str]=None)->dict:
    """Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool."""
    retriever = _get_retriever(thread_id)
    # print(retriever)
    if retriever is None:
        return {"error": "No document found for this thread. Ask user to upload one.", "query": query}
    
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [search_tool, calculator_tool, stock_tool, rag_tool]

toolnode = ToolNode(tools)

llm_tools = llm.bind_tools(tools)


#State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state:ChatState, config:RunnableConfig):
    """LLM node that may answer or request a tool call"""
    thread_id = None
    if config and "configurable" in config:
        thread_id = config.get("configurable", {}).get("thread_id")

    # print(thread_id)
    
    # Dynamic System Message: Tells the AI if a document is actually available
    doc_status = "No document uploaded."
    if thread_id and thread_has_document(thread_id):
        meta = thread_document_metadata(thread_id)
        doc_status = f"Document '{meta['filename']}' is available. Use 'rag_tool' with thread_id '{thread_id}'."

    system_msg = SystemMessage(content=(
        f"You are a helpful assistant. {doc_status} "
        "Use search for news, stock_tool for prices, and rag_tool for PDF questions."
    ))

    messages = [system_msg] + state["messages"]
    response = llm_tools.invoke(messages, config=config)
    return {"messages": [response]}

#Graph
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

def delete_thread(thread_id):
    """Deletes a thread from the summary table and the checkpoint database."""
    try:
        # 1. Delete from your chat summary table
        cursor.execute("DELETE FROM chat WHERE thread_id = ?", (thread_id,))
        
        # 2. Delete from LangGraph's internal checkpoint table
        # LangGraph SqliteSaver uses a table named 'checkpoints' by default
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        
        conn.commit()
        
        # 3. Clean up in-memory RAG data if it exists
        if thread_id in _THREAD_RETRIEVERS:
            del _THREAD_RETRIEVERS[thread_id]
        if thread_id in _THREAD_METADATA:
            del _THREAD_METADATA[thread_id]
            
        return True
    except Exception as e:
        print(f"Error deleting thread: {e}")
        return False


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