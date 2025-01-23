from langchain_core.prompts import ChatPromptTemplate
from llm import get_llama_3dot3_70b_versatile,get_llama3_8b_8192,get_70b_8192
from preprocess_df import full_preprocess
from langchain_multiplayer_tools import ComparePlayerMetrics, ComparePlayerSessions,ComparePlayerSessionsDictionary
from langchain.agents import AgentExecutor, create_tool_calling_agent
import pandas as pd

# # Load your data
# df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-15-17 Players-HIAs-Export_.csv")  # Replace with your data loading logic)  # Replace with your data loading logic
# df = full_preprocess(df)


# # Initialize the tool with your DataFrame
# compare_player_metrics = ComparePlayerMetrics(df=df)

# # Initialize the LLM
# llm = get_llama_3dot3_70b_versatile()

# # Create the prompt template with updated system message
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """You are a sports performance analyst assistant specialising in player metrics.
#         When comparing players:
#         1. Extract ALL player names from the user's query
#         2. Use the compare_player_metrics tool with ALL players mentioned
#         3. Analyze differences between their metrics
#         4. Present insights about how players compare to each other
#         5. Highlight notable patterns or significant differences
        
#         The compare_player_metrics tool can handle any number of players (2 or more),
#         so make sure to include ALL players mentioned in the query.
        
#         When analysing, be thorough in comparing:
#         - Activity distributions
#         - Performance metrics (duration, distance, magnitude)
#         - Metabolic power and stress loads
#         - Time-based patterns
        
#         Present the insights in a clear, comparative format that helps understand
#         the relative performance of all players involved.
#         """
#     ),
#     ("placeholder", "{chat_history}"),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}"),
# ])

# # Create the agent
# agent = create_tool_calling_agent(
#     llm=llm,
#     tools=[compare_player_metrics],
#     prompt=prompt
# )

# # Create the agent executor
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=[compare_player_metrics],
#     verbose=True
# )

# if __name__ == "__main__":
#     # Example comparing multiple players
#     result = agent_executor.invoke({
#         "input": "compare the stats of Lee, Hawk, and Ben"
#     })
#     print(result["output"])











# # ruairi
# # stevie b
# # lee
# # kolo
# # flanno
# # jmce
# # fish
# # rocko
# # hawk
# # beattie
# # j carnegie
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from llm import get_llama_3dot3_70b_versatile, get_llama_3dot1_8b_instant
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Type, Any


def load_session_data(data_directory: str) -> Dict[str, pd.DataFrame]:
    """
    Load multiple session CSV files from a directory.
    
    Args:
        data_directory (str): Path to directory containing session CSV files
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary of DataFrames with session dates as keys
    """
    session_data = {}
    data_path = Path(data_directory)
    
    for csv_file in data_path.glob("*.csv"):
        # Extract date from filename (assuming format includes date)
        # Modify this based on your actual filename format
        date_str = csv_file.stem.split(" ")[0]  # Adjust splitting logic as needed
        
        # Load and preprocess the DataFrame
        df = pd.read_csv(csv_file)
        df = full_preprocess(df)  # Your preprocessing function
        
        session_data[date_str] = df
    
    return session_data

# Load session data
data_directory = r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias"
session_data = load_session_data(data_directory)

# Initialize the tool with your session data
#compare_player_sessions = ComparePlayerSessions(session_data=session_data)
compare_player_sessions_=ComparePlayerSessionsDictionary(session_data=session_data)
# Initialize the LLM
llm = get_llama_3dot1_8b_instant()#get_llama_3dot1_8b_instant()#get_llama3_8b_8192()#get_llama_3dot3_70b_versatile()

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a sports performance analyst assistant specialising in analysing 
        player metrics across different training sessions.
        
        When analysing a player's performance across sessions:
        1. Extract the player name from the user's query
        2. Use the compare_player_sessions tool to analyse their performance
        3. Identify trends and patterns across sessions
        4. Highlight significant changes or improvements
        5. Provide insights about performance consistency and development
        
        Focus on:
        - Changes in activity distribution between sessions
        - Trends in performance metrics
        - Variations in metabolic power and stress loads
        - Time-based pattern changes
        - Overall performance trajectory
        
        Present the insights in a python dictionary.
        
        Your only output will be a python dictionary with the player metrics.

        CRITICAL : aside from the python dictionary, no additional words should be given

        """
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Create the agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=[compare_player_sessions_],
    prompt=prompt
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[compare_player_sessions_],
   #verbose=True
)
# print("---")
# x= agent_executor.invoke({
#         "input": "analyse Lee's performance across all available sessions and provide his metrics as a dictionary"
#     })#["output"]
# print("***")