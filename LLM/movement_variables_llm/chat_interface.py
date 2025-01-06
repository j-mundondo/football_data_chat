import streamlit as st
from api_utils import get_api_response
import random

thinking_phrases = [
"Crunching the numbers...",
"Pondering the possibilities...",
"Contemplating the response...",
"Crafting the reply...",
"Synthesizing the answer..."
]

# In your sidebar or wherever you select the model
if "model" not in st.session_state:
    st.session_state.model = "llama3-8b-8192"  # Set default model

def display_chat_interface():
   # Chat interface
   for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])
   
   # Handle user input
   if prompt := st.chat_input("Ask me anything asbout StatSport variables:"):
       # Add user message to chat history
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"):
           st.markdown(prompt)
       
       # Get API response
       with st.spinner(random.choice(thinking_phrases)):
           response = get_api_response(prompt, st.session_state.session_id, st.session_state.model)
           
           if response:
               # Update session ID and add assistant response
               st.session_state.session_id = response.get('session_id')
               st.session_state.messages.append({"role": "assistant", "content": response['answer']})
               
               # Display assistant response
               with st.chat_message("assistant"):
                   st.markdown(response['answer'])
                   
                   # Show details in expander
                   with st.expander("Details"):
                       st.subheader("Generated Answer")
                       st.code(response['answer'])
                       st.subheader("Model Used")
                       st.code(response['model'])
                       st.subheader("Session ID")
                       st.code(response['session_id'])
           else:
               st.error("Failed to get a response from the API. Please try again.")