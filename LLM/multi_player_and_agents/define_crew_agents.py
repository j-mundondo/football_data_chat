from typing import List, Dict, Optional
import pandas as pd
from crewai_tools import tools
from crewai import Agent, Task, Crew, Process
from LLM.langchain_agent.llm import get_llm
from multiplayer_tools import multi_agent_tools
#print("it worked")



def spin_agents(df):
    llm = get_llm()
    agent_tools = multi_agent_tools(df)
    print(agent_tools)

    dataframe_splitter = Agent(
        llm=llm,
        role="Split Dataframe",
        goal="Segment the dataframe to only include the players relevant to the provided prompt.",
        backstory="A data pre-processor who ensure the data is only centred around the target players.",
        tools=[agent_tools[0]],
        verbose=True,
        code_execution_mode ='unsafe',
        allow_delegation=False)
    
    player_comparison_agent = Agent(
        llm=llm,
        role="Compare the metrics of the target players.",
        goal="Perform pandas operations to retrieve the statistical comparison of the target player.",
        backstory="A analyst who obtains and compares the output of two players and returns them in a well structures, easily understandable manner.",
        code_execution_mode='unsafe',
        tools=[agent_tools[1]],
        verbose=True,
        allow_delegation=False)

    return dataframe_splitter, player_comparison_agent

# df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-12-16 Players-HIAs-Export.csv")
# spin_agents(df)