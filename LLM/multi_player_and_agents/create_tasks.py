from typing import List, Dict, Optional
import pandas as pd
from crewai_tools import tools
from crewai import Agent, Task, Crew, Process
from LLM.langchain_agent.llm import get_llm
from multiplayer_tools import multi_agent_tools
from define_crew_agents import spin_agents

df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-12-16 Players-HIAs-Export.csv") 
dataframe_filterer, player_comparison_agent = spin_agents(df)
split_dataframe, compare_players = multi_agent_tools(df) ## <--------- Check tool vs agent assignment

identify_players = Task(
    description='''
    The user has provided a prompting asking for the comparison of metrics for two or more players.
    To allow subsequent agents to deal with the analysis to compare the players, your job is to 
    filter the dataframe so that is only contain the target players. 
    User prompt: {user_prompt}
    ''',
    expected_output='A dataframe filtered to only the relevant players',
    agent=dataframe_filterer,
    tools=[split_dataframe]
    #,dependencies=
)

compare_players = Task(
    description='''
    Perfrom data analysis on the dataframe to retrieve the relevant answers and comparisons between
    the two target players.  

    User prompt: {user_prompt}
    ''',
    expected_output='Comaprison of relevant metrics between two players',
    agent=player_comparison_agent,
    dependencies=[identify_players],
    context=[identify_players],
    tools=[compare_players]
)

crew = Crew(
    agents=[dataframe_filterer, player_comparison_agent],
    tasks=[
        identify_players, 
        compare_players
        ],
    verbose=True,
    process=Process.sequential
)

user_prompt = 'compare the number of accelerations of Mark and Kolo.'
result = crew.kickoff(inputs={"user_prompt": user_prompt})