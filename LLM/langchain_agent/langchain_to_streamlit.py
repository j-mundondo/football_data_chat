# # import streamlit as st
# # from langchain import hub
# # from langchain.agents import AgentExecutor, create_react_agent, load_tools
# # from langchain_openai import OpenAI
# # from preprocess_df import full_preprocess
# # from create_agent import custom_agent
# # from dotenv import load_dotenv
# # from pathlib import Path
# # import pandas as pd
# # from langchain_community.callbacks.streamlit import (
# #     StreamlitCallbackHandler,
# # )
# # dotenv_path = Path(r'C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\LLM\langchain_agent\.env')
# # load_dotenv(dotenv_path=dotenv_path)
# # logo_url=r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\STATSports-logo.svg"
# # st.set_page_config(
# #     page_title="S.S. Movement Analysis Beta",
# #     page_icon=logo_url,

# # )
# # hide_default_format = """
# #        <style>
# #        #MainMenu {visibility: hidden; }
# #        footer {visibility: hidden;}
# #        </style>
# #        """
# # st.markdown(hide_default_format, unsafe_allow_html=True)
# # # Initialize session state
# # if 'agent' not in st.session_state:
# #     st.session_state.agent = None
# # if 'messages' not in st.session_state:
# #     st.session_state.messages = []

# # # Load environment variables

# # # Set up the Streamlit interface
# # st.image(logo_url, width=100)
# # st.title("Football Movement Analysis ⚽🏃‍♂️")

# # # File uploader
# # uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# # if uploaded_file is not None:
# #     try:
# #         # Process the uploaded file
# #         df = pd.read_csv(uploaded_file)
# #         processed_df = full_preprocess(df)
        
# #         # Create agent if not already created
# #         if st.session_state.agent is None:
# #             st.session_state.agent = custom_agent(processed_df)
# #             st.success("Agent created, ready to answer your query.")
# #     except Exception as e:
# #         st.error(f"Error processing file: {str(e)}")

# # # Chat interface
# # for message in st.session_state.messages:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])

# # if prompt := st.chat_input("What would you like to analyse?"):
# #     if st.session_state.agent is None:
# #         st.error("Please upload a file first!")
# #     else:
# #         # Add user message to chat history
# #         st.session_state.messages.append({"role": "user", "content": prompt})
# #         with st.chat_message("user"):
# #             st.markdown(prompt)

# #         # Get agent response
# #         with st.chat_message("assistant", avatar=logo_url):
# #             st_callback = StreamlitCallbackHandler(st.container())
# #             try:
# #                 response = st.session_state.agent.invoke(
# #                     {"input": prompt}, 
# #                     {"callbacks": [st_callback]}
# #                 )
# #                 st.markdown(response["output"])
# #                 # Add assistant response to chat history
# #                 st.session_state.messages.append({"role": "assistant", "content": response["output"]})
# #             except Exception as e:
# #                 st.error(f"Error getting response: {str(e)}")
# import streamlit as st
# from langchain import hub
# from langchain.agents import AgentExecutor, create_react_agent, load_tools
# from langchain_openai import OpenAI
# from preprocess_df import full_preprocess
# from create_agent import custom_agent
# from dotenv import load_dotenv
# from pathlib import Path
# import pandas as pd
# from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# from langchain.memory import ConversationBufferMemory
# from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

# # Load environment variables
# dotenv_path = Path(r'C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\LLM\langchain_agent\.env')
# load_dotenv(dotenv_path=dotenv_path)

# # Set up Streamlit page config
# logo_url = r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\STATSports-logo.svg"
# st.set_page_config(
#     page_title="S.S. Movement Analysis Beta",
#     page_icon=logo_url,
# )

# # Hide default Streamlit formatting
# hide_default_format = """
#     <style>
#     #MainMenu {visibility: hidden; }
#     footer {visibility: hidden;}
#     </style>
# """
# st.markdown(hide_default_format, unsafe_allow_html=True)

# # Initialize session state
# if 'messages' not in st.session_state:
#     st.session_state.messages = []
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = StreamlitChatMessageHistory()
# if 'memory' not in st.session_state:
#     st.session_state.memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         chat_memory=st.session_state.chat_history
#     )
# if 'agent' not in st.session_state:
#     st.session_state.agent = None

# # Set up the Streamlit interface
# st.image(logo_url, width=100)
# st.title("Football Movement Analysis ⚽🏃‍♂️")

# # Add debug section in sidebar
# with st.sidebar:
#     # Debug Mode Toggle
#     st.session_state.debug_mode = st.toggle("Debug Mode", False)
    
#     # Memory Debug Expander
#     with st.expander("🔍 Debug Memory"):
#         if st.session_state.memory:
#             # Display current memory contents
#             memory_vars = st.session_state.memory.load_memory_variables({})
#             st.write("Current Memory Content:")
#             if "chat_history" in memory_vars:
#                 for i, message in enumerate(memory_vars["chat_history"], 1):
#                     st.text(f"Message {i}:")
#                     st.code(f"Type: {type(message)}\nContent: {message}")
            
#             # Memory inspection tools
#             if st.button("Print Memory Variables"):
#                 st.json(memory_vars)
            
#             if st.button("Print Memory Keys"):
#                 st.write(st.session_state.memory.memory_variables)
    
#     # Clear History Button
#     if st.button("Clear Chat History", type="secondary"):
#         if st.session_state.messages or st.session_state.chat_history:
#             st.session_state.messages = []
#             st.session_state.chat_history.clear()
#             st.session_state.memory.clear()
#             st.success("Chat history and memory cleared!")
#             # Verify memory is cleared
#             memory_vars = st.session_state.memory.load_memory_variables({})
#             st.write("Memory after clearing:", memory_vars)

# # File uploader
# uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# if uploaded_file is not None:
#     try:
#         # Process the uploaded file
#         df = pd.read_csv(uploaded_file)
#         processed_df = full_preprocess(df)
        
#         # Create agent if not already created
#         if st.session_state.agent is None:
#             st.session_state.agent = custom_agent(
#                 dataframe=processed_df,
#                 memory=st.session_state.memory
#             )
#             st.success("Agent created, ready to answer your query.")
#     except Exception as e:
#         st.error(f"Error processing file: {str(e)}")

# # Display chat history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input and response
# if prompt := st.chat_input("What would you like to analyse?"):
#     if st.session_state.agent is None:
#         st.error("Please upload a file first!")
#     else:
#         # Add user message
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Get agent response
#         with st.chat_message("assistant", avatar=logo_url):
#             st_callback = StreamlitCallbackHandler(st.container())
#             try:
#                 # Log memory state before response
#                 if st.session_state.get('debug_mode', False):
#                     st.write("Memory before response:", 
#                             st.session_state.memory.load_memory_variables({}))

#                 response = st.session_state.agent.invoke(
#                     {"input": prompt},
#                     {"callbacks": [st_callback]}
#                 )
#                 st.markdown(response["output"])
                
#                 # Update chat history
#                 st.session_state.messages.append(
#                     {"role": "assistant", "content": response["output"]}
#                 )

#                 # Log memory state after response
#                 if st.session_state.get('debug_mode', False):
#                     st.write("Memory after response:", 
#                             st.session_state.memory.load_memory_variables({}))

#             except Exception as e:
#                 st.error(f"Error getting response: {str(e)}")

import streamlit as st
# from langchain import hub
# from langchain.agents import AgentExecutor, create_react_agent, load_tools
# from langchain_openai import OpenAI
from preprocess_df import full_preprocess
from create_agent import custom_agent
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
#from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# Load environment variables
dotenv_path = Path(r'C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\LLM\langchain_agent\.env')
load_dotenv(dotenv_path=dotenv_path)

# Set up Streamlit page config
logo_url = r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\STATSports-logo.svg"
st.set_page_config(
    page_title="S.S. Movement Analysis Beta",
    page_icon=logo_url,
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

# Set up the main interface
st.image(logo_url, width=100)
st.title("Football Movement Analysis ⚽🏃‍♂️")

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
    with st.chat_message(message["role"]):
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
        with st.chat_message("assistant", avatar=logo_url):
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
                    {"role": "assistant", "content": response["output"]}
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

# Optional: Add session info at the bottom when in debug mode
if st.session_state.get('debug_mode', False):
    with st.expander("Session Info"):
        st.write("Session State Keys:", list(st.session_state.keys()))
        st.write("Agent Status:", "Created" if st.session_state.agent else "Not Created")