# %%
# LIBRARIES AND SET UP
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain.agents import AgentType
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.youtube.search import YouTubeSearchTool
import openai
import pandas as pd
from langchain.agents import Tool
from pathlib import Path
dotenv_path = Path(r'C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\LLM\langchain_agent\.env')
load_dotenv(dotenv_path=dotenv_path)

# %%
llm = ChatGroq(
    api_key=os.environ['GROQ_API_KEY'],  
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)
# %%


# %%
