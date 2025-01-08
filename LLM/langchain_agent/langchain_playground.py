# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import List
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain.agents import AgentType
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.youtube.search import YouTubeSearchTool

# %%
# Load environment variables
load_dotenv()

# Create the Groq LLM instance
llm = ChatGroq(
    api_key='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',  # Explicitly pass the API key
    #model="mixtral-8x7b-32768",  # Use one of Groq's supported models
    model="llama3-8b-8192",#"mixtral-8x7b-32768",  # Use one of Groq's supported models
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


# %%
# Load data
df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\individual_efforts\david_only_metrics.csv")
agent = create_pandas_dataframe_agent(llm, 
                                      df, 
                                      verbose=True,
                                      agent_type="tool-calling",
                                      allow_dangerous_code=True)
# %%
agent.invoke("how many rows are there?")
# %%


# %%
import openai
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools import YouTubeSearchTool

# Create tools list
youtube_tool=YouTubeSearchTool()
tools = [youtube_tool] 
# Load your DataFrame
df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\individual_efforts\david_only_metrics.csv")

# Create the agent with extra tools
PREFIX = "If question is not related to pandas, you can use extra tools. Available tools: YouTube Search"

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    extra_tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
    agent_type="tool-calling"
)

# Test the agent
response = agent.invoke("send me 2 lex friedman videos on youtube")
# %%
print(response)
# %%
