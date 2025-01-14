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
from llm import get_llm
from tools import create_tools
from preprocess_df import full_preprocess
from create_agent import custom_agent
#dotenv_path = Path(r'C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\LLM\langchain_agent\.env')

if __name__ == "__main__":
    dotenv_path = Path(r'C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\LLM\langchain_agent\.env')
    load_dotenv(dotenv_path=dotenv_path)

    # %%
    df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\individual_efforts\david_only_metrics.csv")
    #llm = get_llm()
    #tools=create_tools(df)

    # %%
    df = full_preprocess(df)
    agent = custom_agent(df)

    # %%
    x=agent.invoke("How often do players have to go through a series of high intensity activities and then complete a sprint?")
    display(x)
    # %%
