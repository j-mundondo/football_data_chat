import streamlit as st
from preprocess_df import full_preprocess
from create_agent import custom_agent
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import os
# Setup logo path
logo_path = os.path.join('assets', 'STATSports-logo.png')
LOGO_URL = "https://www.irishfa.com/media/31411/statsports-pod.jpg"
# Set up Streamlit page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="STATSports Movement Analysis Beta",
    page_icon=LOGO_URL#logo_path  # Using emoji instead of file path
)

# Hide default Streamlit formatting
hide_default_format = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_default_format, unsafe_allow_html=True)


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = StreamlitChatMessageHistory()
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=st.session_state.chat_history,
        output_key="output"
    )
if 'agent' not in st.session_state:
    st.session_state.agent = None


st.image(LOGO_URL, width=100)
st.title("Football Movement Analysis (BETA) ⚽")
st.write("This chat app is designed to answer questions regarding high intentisity activities recorded in a football match.")
st.write('''Examples of sample questions include : 
         \n 1. What are the most common 2 event sequences of high-intensity activities within 4 seconds?   
         \n 2. How often do players accelerate and then immediately (within 2 seconds) transition into a sprint?  
         \n 3. How frequently do players decelerate right after completing a sprint?    
         \n 4. What is the most common action after sprints of >20m?    
         \n 5. How often do players have to complete multiple long sprints in a short period?  ''')

# Sidebar Debug Section
with st.sidebar:
    # Debug Mode Toggle
    st.session_state.debug_mode = st.toggle("Debug Mode", False)
    
    # Memory Debug Expander
    with st.expander("🔍 Debug Memory"):
        if st.session_state.memory:
            st.write("Current Memory Content:")
            memory_vars = st.session_state.memory.load_memory_variables({})
            
            if "chat_history" in memory_vars:
                st.markdown("### Message History")
                for i, message in enumerate(memory_vars["chat_history"], 1):
                    st.markdown(f"**Message {i}:**")
                    st.code(f"Type: {type(message)}\nContent: {message}")
                    st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Show Memory Variables"):
                    st.json(memory_vars)
            with col2:
                if st.button("Show Memory Keys"):
                    st.write(st.session_state.memory.memory_variables)
    
    # Clear History Button
    if st.button("Clear Chat History", type="secondary"):
        if st.session_state.messages or st.session_state.chat_history:
            st.session_state.messages = []
            st.session_state.chat_history.clear()
            st.session_state.memory.clear()
            st.success("Chat history and memory cleared!")
            memory_vars = st.session_state.memory.load_memory_variables({})
            st.write("Memory after clearing:", memory_vars)

# File uploader in main area
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Process the uploaded file
        df = pd.read_csv(uploaded_file)
        processed_df = full_preprocess(df)
        
        # Create agent if not already created
        if st.session_state.agent is None:
            st.session_state.agent = custom_agent(
                dataframe=processed_df,
                memory=st.session_state.memory
            )
            st.success("Agent created, ready to answer your query.")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Display existing chat history
for message in st.session_state.messages:
    # with st.chat_message(message["role"]):
    #     st.markdown(message["content"])
    avatar = LOGO_URL if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Chat input and response
if prompt := st.chat_input("What would you like to analyse?"):
    if st.session_state.agent is None:
        st.error("Please upload a file first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        try:
            avatar = LOGO_URL if not os.path.exists(logo_path) else LOGO_URL
            with st.chat_message("assistant", avatar=avatar):
                st_callback = StreamlitCallbackHandler(st.container())
                try:
                    # Show memory before response if in debug mode
                    if st.session_state.get('debug_mode', False):
                        st.markdown("### Memory Before Response")
                        memory_before = st.session_state.memory.load_memory_variables({})
                        if "chat_history" in memory_before:
                            for i, msg in enumerate(memory_before["chat_history"], 1):
                                st.code(f"Message {i}: {msg}")

                    # Get agent response
                    response = st.session_state.agent.invoke(
                        {
                            "input": prompt,
                            "chat_history": st.session_state.memory.load_memory_variables({}).get("chat_history", [])
                        },
                        {"callbacks": [st_callback]}
                    )

                    # Display response
                    st.markdown(response["output"])
                    
                    # Explicitly save context to memory
                    st.session_state.memory.save_context(
                        {"input": prompt},
                        {"output": response["output"]}
                    )
                    
                    # Add to message history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response["output"],"avatar": LOGO_URL}
                    )

                    # Show memory after response if in debug mode
                    if st.session_state.get('debug_mode', False):
                        st.markdown("### Memory After Response")
                        memory_after = st.session_state.memory.load_memory_variables({})
                        if "chat_history" in memory_after:
                            for i, msg in enumerate(memory_after["chat_history"], 1):
                                st.code(f"Message {i}: {msg}")

                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")
                    if st.session_state.get('debug_mode', False):
                        st.error(f"Error type: {type(e).__name__}")
                        st.error(f"Error details: {str(e)}")
        except Exception as e:
            st.error(f"Error with chat display: {str(e)}")

# Optional: Add session info at the bottom when in debug mode
if st.session_state.get('debug_mode', False):
    with st.expander("Session Info"):
        st.write("Session State Keys:", list(st.session_state.keys()))
        st.write("Agent Status:", "Created" if st.session_state.agent else "Not Created")
