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

def format_tool_description(tool):
    """Format tool description for better prompting"""
    return f"""Tool: {tool.name}
Description: {tool.description}
When to use: {tool.when_to_use if hasattr(tool, 'when_to_use') else 'For specific calculations and analysis'}
"""

def create_decision_guide():
    """Create a decision guide for the agent"""
    return """
DECISION PROCESS:
1. Is this a simple frequency/count question?
   → Use DataFrame operations directly (no tools)
2. Does this involve time sequences or patterns?
   → Use appropriate sequence analysis tool
3. Is this a follow-up to previous results?
   → Reference memory/chat history
4. Is this about methodology or concepts?
   → Provide direct explanation

Remember: Tools are for complex analysis ONLY. Simple questions should be answered directly."""

def enhance_tool_descriptions(tools):
    """Add when_to_use field to tools"""
    for tool in tools:
        if not hasattr(tool, 'when_to_use'):
            tool.when_to_use = """
- Use for complex calculations
- Use when time sequences matter
- Use when patterns need analysis
- DON'T use for simple counts or frequencies
"""
    return tools

def custom_agent(dataframe, memory=None):
    """Create a custom agent with enhanced tool guidance"""
    llm = get_llm()
    tools = create_tools(dataframe)
    
    # Enhance tool descriptions
    enhanced_tools = enhance_tool_descriptions(tools)
    
    # Create formatted tool descriptions
    tool_descriptions = "\n".join([format_tool_description(tool) for tool in enhanced_tools])
    
    # Create system prompt with enhanced guidance
    system_prompt = f"""You are a sports movement analyst assistant working with athlete performance data. You have access to a pandas DataFrame and specialized analysis tools.

{create_decision_guide()}

AVAILABLE TOOLS:
{tool_descriptions}

WHEN ANALYZING DATA:
For Basic Questions (DO NOT USE TOOLS):
- Simple counting or frequency questions
- Basic statistics already in the DataFrame
- General observations about the data
- Questions about previous results
- Explanations of concepts

For Complex Analysis (USE TOOLS):
- Time-based sequence analysis
- Multi-event pattern detection
- Complex statistical calculations
- Cross-event relationships
- Detailed performance metrics

DataFrame Information:
{str(dataframe.info())}

Remember: Before using any tool, check if the question can be answered using simple DataFrame operations."""

    # Create the agent with enhanced configuration
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        extra_tools=enhanced_tools,
        verbose=True,
        handle_parsing_errors=True,
        agent_type="openai-tools",
        memory=memory,
        prefix=system_prompt,
        kwargs={
            "max_iterations": 2,
            "early_stopping_method": "generate",
        }
    )
    
    return agent