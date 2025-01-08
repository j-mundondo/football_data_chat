# %%
# BRING IN LIBRARIES
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import BambooLLM
from langchain_groq.chat_models import ChatGroq
from pandasai.connectors import PandasConnector
from pandasai.skills import skill
from pandasai import Agent
from pandasai.ee.vectorstores import ChromaDB
from dotenv import load_dotenv
load_dotenv()
import pandasai as pai
pai.clear_cache()


# %% Field pre-reqs
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
    'Duration_seconds' : 'The number of seconds the High Intensity Activity Type in the row lasted as a float',
    'Long_sprint' : 'A binary flag indicating if a movement is a long sprint. 1 indicates it is a long sprint, 0 means it is not.',
    'Preceding High Intensity Activity Type' : 'The type of high intensity activity type that happened before this row.'
}

# %%
df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\individual_efforts\david_only_metrics.csv")

# %%
# SET UP LLM
llm_api_key=os.environ['GROQ_API_KEY']
langchain_api_key = os.environ['LANGCHAIN_API_KEY']
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    seed=5
)

# %%
# ADD TIME BETWEEN SKILLS QUERY
@skill
def average_time_between_specified_actions(df, action):
    """
    Calculates the average time between specified actions using datetime calculations.
    Call this function when the user askes about the number or average time between
    certain actions.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing player actions
    action (str): The type of action to analyze (e.g., 'Acceleration')
    
    Returns:
    dict: Dictionary containing average time between actions in the required format
    """
    try:
        # Create a copy and filter for specific action
        action_df = df[df['High Intensity Activity Type'] == action].copy()
        
        if len(action_df) < 2:  # Need at least 2 actions to calculate time between them
            return {'type': 'number', 'value': 0.0}
            
        # Convert timestamps to datetime
        action_df['Start Time'] = pd.to_datetime(action_df['Start Time'])
        action_df = action_df.sort_values('Start Time')
        
        # Calculate time differences between consecutive actions
        time_diffs = (action_df['Start Time'] - action_df['Start Time'].shift(1)).dropna()
        
        # Convert to seconds and calculate average
        time_diffs_seconds = time_diffs.dt.total_seconds()
        
        # Filter out any negative or zero differences
        time_diffs_seconds = time_diffs_seconds[time_diffs_seconds > 0]
        
        if len(time_diffs_seconds) > 0:
            average_time = round(time_diffs_seconds.mean(), 2)
            return {
                'type': 'number',
                'value': float(average_time)
            }
        else:
            return {
                'type': 'number',
                'value': 0.0
            }
            
    except Exception as e:
        print(f"Error in average_time_between_actions: {str(e)}")
        return {
            'type': 'number',
            'value': 0.0
        }

@skill
def number_of_occurences_of_action_in_timespan(df, action, time_in_seconds):
    """
    Identifies occurrences of multiple actions within a specified timespan in seconds.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing player actions
    action (str): The type of action to analyze (e.g., 'Acceleration')
    time_in_seconds (int): The time window to check for multiple actions
    
    Returns:
    dict: Formatted response with count of windows containing multiple actions
    """
    # Convert Start Time to datetime if not already
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    
    # Filter for specific action
    action_df = df[df['High Intensity Activity Type'] == action].copy()
    action_df = action_df.sort_values('Start Time')
    
    # Initialize variables for tracking windows
    windows = []
    current_window = []
    
    # Analyze each action
    for idx, row in action_df.iterrows():
        current_time = row['Start Time']
        
        # If window is empty, add first action
        if not current_window:
            current_window = [(current_time, row)]
            continue
        
        # Check if current action is within timespan of window start
        time_diff = (current_time - current_window[0][0]).total_seconds()
        
        if time_diff <= time_in_seconds:
            # Add to current window if within timespan
            current_window.append((current_time, row))
        else:
            # If current window has multiple actions, save it
            if len(current_window) > 1:
                windows.append(current_window)
            # Start new window with current action
            current_window = [(current_time, row)]
    
    # Add final window if it has multiple actions
    if len(current_window) > 1:
        windows.append(current_window)
    
    # Calculate statistics
    total_windows = len(windows)
    
    return {
        'type': 'number',
        'value': float(total_windows)
    }

# %% 
#multiple_actions_in_a_timespan(df, 'Sprint', 60)

#%%
# AGENT SET UP
vector_store = ChromaDB()
agent = Agent(df, vectorstore=vector_store,memory_size=0, config={
        "field_descriptions": field_descriptions,
        'llm': llm,
        "verbose": True,
        "custom_whitelisted_dependencies": ["dateutil"]})  # Helpful for debugging)
agent.add_skills(average_time_between_specified_actions)#,number_of_occurences_of_action_in_timespan)
#agent.add_skills(multiple_actions_in_a_timespan)
 

# %%
# SET UP TRAINING DATA
# query = "What is the average time between accelerations"
# response = '''
# import pandas as pd
# df = dfs[0]
# result = average_time_between_actions(df, 'Acceleration')
# '''

# # Training data setup - one example per skill
# agent.train(queries=[query], codes=[response])

# # %% 
# # Chat with the agent
# response = agent.chat("how much time passes between sprints?")
# print(response)

# # %%
# response = agent.chat("How many times were there multiple decelration within 1 minute?")
# print(response)
# %%
# %% 
# Chat with the agent
response = agent.chat("What is the average time between sprints?")
print(response)
# %%
print(agent.last_code_generated)
# %%
response2 = agent.chat("How many times were there multiple decelration within 1 minute?")
print(response2)
# %%
print(agent.last_code_generated)

# %%
response3 = agent.chat("How many rows are there in the dataframe?")
print(response3)

# %%
print(agent.last_code_generated)

# %%
