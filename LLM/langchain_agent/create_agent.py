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
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def custom_agent(dataframe, memory=None):
    """Create a custom agent optimized for Groq"""
    llm = get_llm()
    tools = create_tools(dataframe)
    
    # Simplified prompt that's more compatible with Groq
#     system_prompt = """You are a sports movement analyst. You have access to a pandas DataFrame with athlete performance data.

# Simple Rules:
# 1. For basic counts and frequencies, use DataFrame directly without tools
# 2. Use tools only for:
#    - Time sequences
#    - Complex patterns
#    - Advanced statistics
# 3. Start each response by deciding if tools are needed

# Examples:
# - "What's the most common action?" → No tools needed
# - "What sequences occur in 5 seconds?" → Use tools
# - "How many sprints?" → No tools needed
# - "Do sprints follow accelerations?" → Use tools

# Current Data Summary:
# {df_info}

# Please note that the above is only a representatitve set using df.head(). So for question that don't require tools e.g. how many
# rows in the data you can't only consider the 5 rows you see. you still have to do something to the effect of len(df).

# Remember: Keep it simple. Only use tools when absolutely necessary."""

    system_prompt = """You are a sports movement analyst. You have access to a pandas DataFrame with athlete performance data.

    Simple Rules:
    1. For basic counts and frequencies, ALWAYS use the full DataFrame (df) not just the preview
    2. Key DataFrame operations:
    - Basic counts: Use len(df) for total rows
    - Value counts: Use df['column'].value_counts()
    - Summary stats: Use df.describe()
    3. Use tools only for:
    - Time sequences
    - Complex patterns
    - Advanced statistics

    Data Context:
    Total rows in dataset: {total_rows}
    Preview of data structure:
    {df_info}

    IMPORTANT: While you only see a preview above, you must ALWAYS operate on the full dataset.
    For example:
    - ❌ Wrong: Counting rows in preview
    - ✅ Right: Using len(df) for total count
    - ❌ Wrong: df.head()['column'].value_counts()
    - ✅ Right: df['column'].value_counts()

    Remember: Any operation involving counts, frequencies, or statistics must be performed on the entire DataFrame, not just the preview."""

    # Create the agent with simplified configuration
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        #number_of_head_rows=dataframe.shape[0],
        extra_tools=tools,
        verbose=True,
        handle_parsing_errors=True,
         allow_dangerous_code=True,
        agent_type="tool-calling",
        memory=memory,
        prefix=system_prompt.format(df_info=str(dataframe.info())),
    )
    
    return agent