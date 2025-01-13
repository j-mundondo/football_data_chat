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
    model="llama3-8b-8192",#"Llama-3-Groq-8B-Tool-Use",#"Llama-3-Groq-70B-Tool-Use",##"mixtral-8x7b-32768",  # Use one of Groq's supported models
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

from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
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
# Let's inspect some of the attributes associated with the tool.
print(multiply.name)
print(multiply.description)
print(multiply.args)
# %%
from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

print(calculator.invoke({"a": 2, "b": 3}))
print(await calculator.ainvoke({"a": 2, "b": 5}))
# %%
from langchain_core.tools import tool
from datetime import datetime
import pandas as pd

@tool
def calculate_time_between_actions(action: str) -> str:
    """
    Calculate average time between specified actions (sprints, accelerations, decelerations).
    Args:
        action: Type of action to analyze ('Sprint', 'Acceleration', 'Deceleration')
    Returns:
        String describing the average time between actions
    """
    try:
        # Use the df from the outer scope (the one loaded in your agent)
        action_df = df[df['High Intensity Activity Type'] == action].copy()
        
        if len(action_df) < 2:
            return f"Not enough {action} events found to calculate time between them."
            
        action_df['Start Time'] = pd.to_datetime(action_df['Start Time'])
        action_df = action_df.sort_values('Start Time')
        
        time_diffs = (action_df['Start Time'] - action_df['Start Time'].shift(1)).dropna()
        time_diffs_seconds = time_diffs.dt.total_seconds()
        time_diffs_seconds = time_diffs_seconds[time_diffs_seconds > 0]
        
        if len(time_diffs_seconds) > 0:
            average_time = round(time_diffs_seconds.mean(), 2)
            return f"The average time between {action}s is {average_time} seconds."
        else:
            return f"No valid time differences found between {action} events."
            
    except Exception as e:
        return f"Error calculating time between {action}s: {str(e)}"

# Create more sports-specific tools
@tool
def analyze_intensity_distribution(column_name: str, bins: int = 5) -> str:
    """
    Analyze the distribution of values in a specified intensity metric column.
    Args:
        column_name: Name of the column to analyze (e.g., 'MaxSpeed', 'Acceleration')
        bins: Number of bins for distribution analysis
    """
    try:
        distribution = pd.qcut(df[column_name], q=bins, duplicates='drop')
        counts = distribution.value_counts().sort_index()
        
        result = f"Distribution analysis for {column_name}:\n"
        for interval, count in counts.items():
            result += f"- Range {interval}: {count} occurrences\n"
        return result
    except Exception as e:
        return f"Error analyzing distribution: {str(e)}"

# %%
# Add tools to your agent
tools = [
    youtube_tool,
    calculate_time_between_actions,
    analyze_intensity_distribution
]

# Create agent with new tools
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    extra_tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
    agent_type="tool-calling"
)

# Test the tools
responses = [
    agent.invoke("What's the average time between sprints?"),
    agent.invoke("What's the average time between accelerations?"),
    # You can also combine tools with DataFrame operations
    agent.invoke("Compare the time between sprints in the first and second half of the session")
]

# %%
for response in responses:
    print(response)
# %%
