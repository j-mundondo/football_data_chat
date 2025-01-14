# from llm import get_llm
# from tools import create_tools
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# llm = get_llm()

# from llm import get_llm
# from tools import create_tools
# import pandas as pd
# from langchain.prompts import PromptTemplate

# from langchain.prompts import PromptTemplate

# PREFIX = """You are an expert sports movement analyst assistant working with athlete performance data. You have access to a pandas DataFrame containing movement and performance data. When analyzing this data:

# IMPORTANT GUIDELINES FOR USING TOOLS:
# 1. For simple frequency questions or pattern observations, use the pandas DataFrame directly without external tools
# 2. Only use specialized tools when you need to:
#    - Calculate complex sequences
#    - Analyze time-based patterns
#    - Perform multi-step calculations
#    - Generate detailed statistics
# 3. For questions about what appears most often or basic patterns:
#    - Use basic DataFrame operations (value_counts, groupby, etc.)
#    - DON'T call external tools unless time sequences or complex patterns are needed

# HOW TO RESPOND:
# 1. First, determine if the question needs tools:
#    - Simple frequency or counts → Use DataFrame directly
#    - Complex sequences or time patterns → Use tools
#    - General questions → Use available data directly

# 2. If using DataFrame directly:
#    - Read the data
#    - Provide the insight
#    - Explain what you found

# 3. If using tools:
#    - Explain which tool you're using
#    - Show the results
#    - Interpret the findings

# Remember: Many questions can be answered by simply looking at the data - don't overcomplicate your analysis by using tools when basic DataFrame operations would suffice.

# Current Conversation Context:
# {chat_history}"""

# def create_agent_prompt():
#     """Creates the prompt template for the agent"""
#     return PromptTemplate(
#         template=PREFIX,
#         input_variables=["chat_history", "input"]
#     )
# suffix = ''

# # def custom_agent(dataframe, memory=None):
# #     """
# #     Create a custom agent with optional memory integration
# #     Args:
# #         dataframe: pandas DataFrame to analyze
# #         memory: optional memory instance for conversation history
# #     """
# #     llm = get_llm()
# #     tools = create_tools(dataframe)
# #     prompt = create_agent_prompt()
    
# #     agent = create_pandas_dataframe_agent(
# #         llm=llm,
# #         df=dataframe,
# #         prefix=PREFIX,
# #         extra_tools=tools,
# #         verbose=True,
# #         handle_parsing_errors=True,
# #         allow_dangerous_code=True,
# #         agent_type="tool-calling",
# #         memory=memory  # Pass memory directly to agent creation
# #     )
    
# #     return agent
from llm import get_llm
from tools import create_tools
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def custom_agent(dataframe, memory=None):
    """Create a minimal custom agent for Groq"""
    llm = get_llm()
    tools = create_tools(dataframe)
    
    system_prompt = """You analyze sports movement data. Use tools only for time-based or sequence analysis. For simple counts, use DataFrame directly.

Example: 
"Most common action?" → No tools
"Action sequences?" → Use tools"""

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        extra_tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        agent_type="openai-tools",
        memory=memory,
        prefix=system_prompt
    )
    
    return agent