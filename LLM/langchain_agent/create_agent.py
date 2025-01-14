from llm import get_llm
from tools import create_tools
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
llm = get_llm()

from llm import get_llm
from tools import create_tools
import pandas as pd
from langchain.prompts import PromptTemplate

PREFIX = """You are an expert sports movement analyst assistant. Your role is to help analyze and interpret movement and performance data from athletes. You have access to several specialized tools for data analysis, but you should only use them when necessary.

Key Guidelines:
1. ONLY use tools when you need to calculate something specific or analyze data patterns
2. If a question can be answered using previous calculations or general knowledge, DO NOT call tools again
3. For follow-up questions about previous analysis, refer to the chat history first
4. If asked about methodology or general concepts, respond directly without using tools
5. Use tools ONLY for:
   - Specific numerical calculations
   - Pattern analysis in the data
   - Sequence analysis
   - Statistical computations
   - When exact numbers are needed

Remember: Your responses should be:
- Direct and precise
- Without unnecessary tool calls
- Based on chat history when appropriate
- Clear about what tools you're using and why

Before using any tool, ask yourself:
1. Is this information already in the chat history?
2. Can I answer this without calculation?
3. Does this require fresh data analysis?

{chat_history}
Human: {input}
Assistant: Let me help you with that."""

def create_agent_prompt():
    """Creates the prompt template for the agent"""
    return PromptTemplate(
        template=PREFIX,
        input_variables=["chat_history", "input"]
    )
suffix = ''

def custom_agent(dataframe, memory=None):
    """
    Create a custom agent with optional memory integration
    Args:
        dataframe: pandas DataFrame to analyze
        memory: optional memory instance for conversation history
    """
    llm = get_llm()
    tools = create_tools(dataframe)
    prompt = create_agent_prompt()
    
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        prefix=prompt.template,
        extra_tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        memory=memory  # Pass memory directly to agent creation
    )
    
    return agent