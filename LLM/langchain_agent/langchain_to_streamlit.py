import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools
from langchain_openai import OpenAI
from preprocess_df import full_preprocess
from create_agent import custom_agent
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)
dotenv_path = Path(r'C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\LLM\langchain_agent\.env')
load_dotenv(dotenv_path=dotenv_path)
logo_url=r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\STATSports-logo.svg"
st.set_page_config(
    page_title="S.S. Movement Analysis Beta",
    page_icon=logo_url,

)
hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)
# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Load environment variables

# Set up the Streamlit interface
st.image(logo_url, width=100)
st.title("Football Movement Analysis ⚽🏃‍♂️")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Process the uploaded file
        df = pd.read_csv(uploaded_file)
        processed_df = full_preprocess(df)
        
        # Create agent if not already created
        if st.session_state.agent is None:
            st.session_state.agent = custom_agent(processed_df)
            st.success("Agent created, ready to answer your query.")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to analyse?"):
    if st.session_state.agent is None:
        st.error("Please upload a file first!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant", avatar=logo_url):
            st_callback = StreamlitCallbackHandler(st.container())
            try:
                response = st.session_state.agent.invoke(
                    {"input": prompt}, 
                    {"callbacks": [st_callback]}
                )
                st.markdown(response["output"])
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")