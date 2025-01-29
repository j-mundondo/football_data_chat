from langchain_groq import ChatGroq
import os

api_key=os.getenv('GROQ_API_KEY')
def get_llama3_8b_8192():
    """
    Create and return a Groq LLM instance using Streamlit secrets for API key
    """
    try:
        # Create the Groq LLM instance using Streamlit secrets
        llm = ChatGroq(
            api_key=api_key,
            model="llama3-8b-8192",
           # model="groq/llama3-70b-8192",#
            #model="groq/llama-3.3-70b-versatile", #<----------
            #model="groq/llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=None, 
            timeout=None,
            max_retries=2
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        print("Please ensure GROQ_API_KEY is set in your Streamlit secrets.")
        raise

def get_70b_8192():
    """
    Create and return a Groq LLM instance using Streamlit secrets for API key
    """
    try:
        # Create the Groq LLM instance using Streamlit secrets
        llm = ChatGroq(
            api_key=api_key,
            #model="llama3-8b-8192",
            model="llama3-70b-8192",#
            #model="groq/llama-3.3-70b-versatile",
            #model="groq/llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=None, 
            timeout=None,
            max_retries=2
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        print("Please ensure GROQ_API_KEY is set in your Streamlit secrets.")
        raise

def get_llama_3dot3_70b_versatile():
    """
    Create and return a Groq LLM instance using Streamlit secrets for API key
    """
    try:
        # Create the Groq LLM instance using Streamlit secrets
        llm = ChatGroq(
            api_key=api_key,
            #model="llama3-8b-8192",
            #model="groq/llama3-70b-8192",#
            model="llama-3.3-70b-versatile",
            #model="groq/llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=None, 
            timeout=None,
            max_retries=2
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        print("Please ensure GROQ_API_KEY is set in your Streamlit secrets.")
        raise

def get_llama_3dot1_8b_instant():
    """
    Create and return a Groq LLM instance using Streamlit secrets for API key
    """
    try:
        # Create the Groq LLM instance using Streamlit secrets
        llm = ChatGroq(
            api_key=api_key,
            #model="llama3-8b-8192",
            #model="groq/llama3-70b-8192",#
            #model="groq/llama-3.3-70b-versatile",
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=None, 
            timeout=None,
            max_retries=2
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        print("Please ensure GROQ_API_KEY is set in your Streamlit secrets.")
        raise

def DeepSeek():
    """
    Create and return a Groq LLM instance using Streamlit secrets for API key
    """
    try:
        # Create the Groq LLM instance using Streamlit secrets
        llm = ChatGroq(
            api_key=api_key,
            #model="llama3-8b-8192",
            #model="groq/llama3-70b-8192",#
            #model="groq/llama-3.3-70b-versatile",
            model="DeepSeek-R1-Distill-Llama-70b",
            temperature=0.0,
            max_tokens=None, 
            timeout=None,
            max_retries=2
        )
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        print("Please ensure GROQ_API_KEY is set in your Streamlit secrets.")
        raise
