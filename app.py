import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from lggraph.langraph import graph
from dotenv import dotenv_values

config = dotenv_values(".env")
psswrd = config["PASSWORD"]

# app config
st.set_page_config(page_title="News AI", page_icon="🤖")
st.title("News AI 📰")
st.info('The infos may not be fully accurate. Please verify with original sources.', icon="ℹ️")

def main():
    chinput = False # Controls the chat_input whether to show or not if the password is correct
    with st.sidebar:
        password = st.text_input("Enter the password to continue", type="password")
        if password == psswrd:
            chinput = True

    welcome_message = "Hello! What can I help with? 😊"
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content=welcome_message),
        ]

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                # typewriter(text=welcome_message, speed=17)
                # st.write("asd")
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    # user input
    if chinput == True:
        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.markdown(
                """
            <style>
                .st-emotion-cache-janbn0 {
                    flex-direction: row-reverse;
                    text-align: right;
                }
            </style>
            """,
                unsafe_allow_html=True,
            )
            # st.write(st.session_state.chat_history)

            with st.chat_message("Human"):
                st.markdown(user_query) 
                

            with st.chat_message("AI"):
                with st.spinner("please wait..."):

                    response = graph(user_query, st.session_state.chat_history)
                    log_response = response["intermediate_steps"][-1].log 
                st.write(log_response)
                # st.write(list(response))
                # st.write(response)

            st.session_state.chat_history.append(AIMessage(content=log_response))

            # st.write(st.session_state.chat_history)

if __name__ == "__main__":
    main()