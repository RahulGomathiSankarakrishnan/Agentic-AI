from Backend_3_Tools import chatbot, create_table, retrieve_all_threads, retrieve_all_summaries, save_summary
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid

# --- CONFIG & LLM ---
st.set_page_config(page_title="AI Chatbot", layout="wide")
llm = ChatOpenAI() # Use gpt-4o or your preferred model model="gpt-4o"
create_table()
# --- UTILITY FUNCTIONS ---
def generate_thread():
    return str(uuid.uuid4())    

def reset_chat():
    new_id = generate_thread()
    st.session_state['thread_id'] = new_id
    # if new_id not in st.session_state['chat_threads']:
    #     st.session_state['chat_threads'].append(new_id)
    st.session_state['message_history'] = []
    # st.rerun()

def add_thread_to_sidebar(thread_id):
    """Call this only when the first message is sent."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_thread(thread_id):
    """Loads history for a specific thread from the LangGraph backend."""
    st.session_state['thread_id'] = thread_id
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])
    
    formatted_history = []
    for msg in messages:
# Only include Human and AI messages; skip ToolMessage or messages with tool_calls
        if isinstance(msg, HumanMessage):
            formatted_history.append({'role': "user", 'content': msg.content})
        elif isinstance(msg, AIMessage):
            # Only add if it's a final response (has content and no tool calls)
            if msg.content.strip(): 
                formatted_history.append({'role': "assistant", 'content': msg.content})
    
    st.session_state['message_history'] = formatted_history

def get_summary(thread_id):
    """Finds the summary for a thread or returns a placeholder."""
    return next((item['summary'] for item in st.session_state['summary_history'] 
                if item['thread_id'] == thread_id), "New Conversation...")

def run_summarize(thread_id, history, config):
    """Generates a short title for the chat based on the first few messages."""
    if not any(s['thread_id'] == thread_id for s in st.session_state['summary_history']):
        if len(history) >= 1:
            text_to_summarize = history[0]['content']
            prompt = f"Generate the topic of this chat in 3-5 words: {text_to_summarize}"
            summary = llm.invoke(prompt,config).content.replace('"', '')
            st.session_state['summary_history'].append({'thread_id': thread_id, 'summary': summary})
            save_summary(thread_id,summary)

# --- SESSION STATE INITIALIZATION ---
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread()
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
if 'summary_history' not in st.session_state:
    st.session_state['summary_history'] = retrieve_all_summaries()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🤖 GPT Assistant")
    if st.button("➕ New Chat", use_container_width=True) and len(st.session_state['message_history']) > 0:
        reset_chat()
        st.rerun()
    
    st.divider()
    st.subheader("Recent Conversations")
    
    # Render threads in reverse chronological order
    for item in st.session_state['summary_history']:
        t_id = item['thread_id']
        title = item['summary']
        # Highlight active thread
        is_active = "✅ " if t_id == st.session_state['thread_id'] else ""
        if st.button(f"{is_active}{title}", key=f"btn_{t_id}", use_container_width=True):
            load_thread(t_id)
            st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title(f"Chat: {get_summary(st.session_state['thread_id'])}")

# Display Chat History
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat Input
user_input = st.chat_input("Message GPT...")

if user_input:
    add_thread_to_sidebar(st.session_state['thread_id'])
    # 1. User Message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. Assistant Response (Streaming)
    with st.chat_message("assistant"):
        config = {'configurable': {'thread_id': st.session_state['thread_id']},'run_name':'chat_turn','metadata':{'thread_id':st.session_state['thread_id']}}
        config_summary = {'configurable': {'thread_id': st.session_state['thread_id']},'run_name':'topic_summary','metadata':{'thread_id':st.session_state['thread_id']}}
        status_holder = {"box": None}
        # Generator for streaming
        def stream_response():
            for chunk, metadata in chatbot.stream(
                {'messages': HumanMessage(content=user_input)}, 
                config=config, 
                stream_mode='messages'
            ):
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
                if isinstance(chunk,AIMessage):
                    yield chunk.content
        full_response = st.write_stream(stream_response())
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )
    # 3. Update State
    st.session_state['message_history'].append({'role': 'assistant', 'content': full_response})
    
    # 4. Auto-Summarize if this is the first exchange
    run_summarize(st.session_state['thread_id'], st.session_state['message_history'], config_summary)
    st.session_state['summary_history'] = retrieve_all_summaries()
    st.rerun()
