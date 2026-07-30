import streamlit as st

# with st.chat_message('user'):
#     st.text('Hi')

# with st.chat_message('assistant'):
#     st.text('How can I help you?')

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]  #To store messages at every chat input entry - every entry means program will

#sidebar
st.sidebar.title('LangGraph Chatbot')
st.sidebar.button('New Chat')
st.sidebar.header('My Conversations')

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    with st.chat_message('user'):
        st.text(user_input)
        st.session_state['message_history'].append({'role':'user','content':user_input})

    with st.chat_message('assistant'):
        st.text(user_input)
        st.session_state['message_history'].append({'role':'assistant','content':user_input})

