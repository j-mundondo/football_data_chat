from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pathlib import Path
dotenv_path = Path(r'C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\LLM\langchain_agent\.env')
load_dotenv(dotenv_path=dotenv_path)

def get_llm():
    # Create the Groq LLM instance
    llm = ChatGroq(
        api_key=os.environ['GROQ_API_KEY'],  
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2)
    return llm