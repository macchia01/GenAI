import streamlit as st
from utils.QnA import Q_A
import re,time


def QA_Bot(vectorstore):
    st.title("Q&A Bot")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        ai_response = Q_A(vectorstore,prompt)
        response = f"Echo: {ai_response}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in re.split(r'(\s+)', response):
                full_response += chunk + " "
                time.sleep(0.01)

                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})