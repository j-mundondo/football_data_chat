from langchain_groq import ChatGroq
import streamlit as st

def get_llm():
    """
    Create and return a Groq LLM instance using Streamlit secrets for API key
    """
    try:
        # Create the Groq LLM instance using Streamlit secrets
        llm = ChatGroq(
            api_key=st.secrets["GROQ_API_KEY"],
            #model="llama3-8b-8192",
            #model="llama3-70b-8192",#
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=None, 
            timeout=None,
            max_retries=4
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        st.error("Please ensure GROQ_API_KEY is set in your Streamlit secrets.")
        raise