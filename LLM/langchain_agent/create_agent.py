from llm import get_llm
from tools import create_tools
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
llm = get_llm()
#tools=create_tools()

# def custom_agent(dataframe=None):
#     tools=create_tools()
#     agent = create_pandas_dataframe_agent(
#     llm=llm,
#     df=dataframe,
#     extra_tools=tools,
#     verbose=True, 
#     handle_parsing_errors=True,
#     allow_dangerous_code=True,
#     agent_type="tool-calling"
#     )
#     return agent

from llm import get_llm
from tools import create_tools
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

def custom_agent(dataframe, memory=None):
    """
    Create a custom agent with optional memory integration
    Args:
        dataframe: pandas DataFrame to analyze
        memory: optional memory instance for conversation history
    """
    llm = get_llm()
    tools = create_tools(dataframe)
    
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        extra_tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        memory=memory  # Pass memory directly to agent creation
    )
    
    return agent