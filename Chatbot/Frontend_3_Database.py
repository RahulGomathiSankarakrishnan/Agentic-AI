from Backend_1_original import chatbot
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import streamlit as st
import uuid

llm = ChatOpenAI()

#Utility fns
def generate_thread():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread()
    st.session_state['thread_id']=thread_id
    add_thread(st.session_state['thread_id'])
    new = True
    del st.session_state.message_history[:]
    st.rerun()

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable':{'thread_id':thread_id}})
    return state.values.get('messages',[])

def summarize(thread_id, prompts):
    if st.session_state['thread_id'] not in st.session_state['summary_history']:
        prompt = f"For the provided conversation, generate the topic of the messages within 5 words to be used as it's reference.\n{prompts}"
        summary = llm.invoke(prompt).content
        st.session_state['summary_history'].append({'thread_id':thread_id,'summary':summary})

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

if 'summary_history' not in st.session_state:
    st.session_state['summary_history'] = []

add_thread(st.session_state['thread_id'])

#Sidebar
st.sidebar.title('LangGraph Chatbot')
if st.sidebar.button('New Chat') and len(st.session_state['message_history'])>0:
    reset_chat()
st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    current_summary = next((item.get("summary") for item in st.session_state['summary_history'] if item.get("thread_id")==thread_id),"Blank")
    if st.sidebar.button(str(current_summary)):
        if len(st.session_state['message_history'])==0:
            st.session_state['chat_threads'].remove(st.session_state['thread_id'])            
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_dict=[]
        for message in messages:
            if isinstance(message, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_dict.append({'role':role,'content':message.content})
        st.session_state['message_history'] = temp_dict
        
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    CONFIG = {'configurable':{'thread_id':st.session_state['thread_id']}}

    with st.chat_message('assistant'):
        ai_message=st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {'messages':HumanMessage(content=user_input)},config=CONFIG,stream_mode='messages'
            ))
    st.session_state['message_history'].append({'role':'assistant','content':ai_message})
    summarize(st.session_state['thread_id'],st.session_state['message_history'])
