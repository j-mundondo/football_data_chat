# BRING IN LIBRARIES
# %%
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import BambooLLM
from langchain_groq.chat_models import ChatGroq
from pandasai.connectors import PandasConnector
from pandasai import Agent
from pandasai.ee.vectorstores import ChromaDB
from pandasai.ee.agents.semantic_agent import SemanticAgent
from pandasai.ee.agents.judge_agent import JudgeAgent
from dotenv import load_dotenv
load_dotenv()
import pandasai as pai
pai.clear_cache()

# Field pre-reqs (keep your existing field descriptions)
field_descriptions = {
    'High Intensity Activity Type': 'Classification of the movement (Acceleration/Deceleration/Sprint)',
    'Start Time': 'Timestamp when activity began (HH:MM:SS)',
    'End Time': 'Timestamp when activity ended (HH:MM:SS)', 
    'Time Since Last': 'Duration since previous activity of that same type (MM:SS.S)',
    'Duration': 'Length of activity (MM:SS.S)',
    'Distance': 'Meters traveled during activity',
    'Magnitude': 'Peak intensity measure of the activity',
    'Avg Metabolic Power': 'Average energy expenditure during activity (W/kg)',
    'Dynamic Stress Load': 'Cumulative stress metric from activity intensity',
    'Duration_seconds': 'The number of seconds the High Intensity Activity Type in the row lasted as a float',
    'Long_sprint': 'A binary flag indicating if a movement is a long sprint. 1 indicates it is a long sprint, 0 means it is not.',
    'Preceding High Intensity Activity Type': 'The type of high intensity activity type that happened before this row.'
}

# Read data
df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\individual_efforts\david_only_metrics.csv")

# Set up LLM (keep your existing setup)
llm_api_key = os.environ['GROQ_API_KEY']
langchain_api_key = os.environ['LANGCHAIN_API_KEY']
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Set up agent
vector_store = ChromaDB()
agent = Agent(df, vectorstore=vector_store, 
              description="You are a data analysis agent. Your goal is to answer the questions asked logically using flawless code to arrive at an answer",
              config={
    "field_descriptions": field_descriptions,
    'llm': llm,
    "verbose": True,
    "custom_whitelisted_dependencies": ["dateutil"]
})

# Training data for average time between actions
time_between_query = "What is the average time between accelerations?"
time_between_code = """
import pandas as pd

df = dfs[0]
action = 'Acceleration'

# Create a copy and filter for specific action
action_df = df[df['High Intensity Activity Type'] == action].copy()

# Convert timestamps to datetime
action_df['Start Time'] = pd.to_datetime(action_df['Start Time'])
action_df = action_df.sort_values('Start Time')

# Calculate time differences between consecutive actions
time_diffs = (action_df['Start Time'] - action_df['Start Time'].shift(1)).dropna()

# Convert to seconds and calculate average
time_diffs_seconds = time_diffs.dt.total_seconds()

# Filter out any negative or zero differences
time_diffs_seconds = time_diffs_seconds[time_diffs_seconds > 0]

average_time = round(time_diffs_seconds.mean(), 2)
result = {
    'type': 'number',
    'value': float(average_time)
}
"""

# Training data for multiple actions in timespan
multiple_actions_query = "How many times were there multiple decelerations within 1 minute?"
multiple_actions_code = """
import pandas as pd

df = dfs[0]
action = 'Deceleration'
time_in_seconds = 60

# Convert Start Time to datetime
df['Start Time'] = pd.to_datetime(df['Start Time'])

# Filter for specific action and sort
action_df = df[df['High Intensity Activity Type'] == action].copy()
action_df = action_df.sort_values('Start Time')

windows = []
current_window = []

for idx, row in action_df.iterrows():
    current_time = row['Start Time']
    
    if not current_window:
        current_window = [(current_time, row)]
        continue
    
    time_diff = (current_time - current_window[0][0]).total_seconds()
    
    if time_diff <= time_in_seconds:
        current_window.append((current_time, row))
    else:
        if len(current_window) > 1:
            windows.append(current_window)
        current_window = [(current_time, row)]

if len(current_window) > 1:
    windows.append(current_window)

result = {
    'type': 'number',
    'value': float(len(windows))
}
"""

# Train the agent
agent.train(
    queries=[time_between_query, multiple_actions_query],
    codes=[time_between_code, multiple_actions_code]
)

# %%
# Now you can chat with the agent
query="What is the time between accelerations?"
rephrased_query = agent.rephrase_query(query)
response = agent.chat(rephrased_query)
print(response)

# %%

rephrased_query = agent.rephrase_query(query)
print("The rephrased query is", rephrased_query)

# %%
questions = agent.clarification_questions(query)
for question in questions:
    print(question)

# %%
# Explain how the chat response is generated
response = agent.explain()
print(response)

# %%
print(agent.last_code_generated)

# %%
response = agent.chat('What are the different High Intensity Activity Types')
print(response)

# %%
# response2 = agent.chat("How many times were there multiple decelerations within 1 minute?")
# print(response2)

# # %%
# response3 = agent.chat("How many rows are there in the data?")
# print(response3)
# # %%
# print(agent.last_code_generated)