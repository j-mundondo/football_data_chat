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
# SET UP LLM
llm_api_key=os.environ['GROQ_API_KEY']
#bamboo_llm = BambooLLM(api_key='$2a$10$x8bhQ4IEibvjy1zxxADjyehGbujFf9Yw.HTusHZoIS6bpGa4Dbfee')
langchain_api_key = os.environ['LANGCHAIN_API_KEY']

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# %%
#read in data
df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\individual_efforts\david_only_metrics.csv")
df = SmartDataframe(df,config = {'llm':llm, 'verbose':True})

# %%
df.chat('Find the average of the column named Dynamic Stress Load')
print(df.last_code_executed)

# %%
# bamboo_df = SmartDataframe(data,config = {'llm':bamboo_llm})
# response = bamboo_df.chat("Plot dynamic stress load on a line graph")
#print(response)

# %%
# DF WITH CONNECTOR

df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\individual_efforts\david_only_metrics.csv")
connector = PandasConnector({"original_df": df}, field_descriptions=field_descriptions)
sdf = SmartDataframe(connector)
#sdf.chat("What are the columns in the DF?")

# %%
import pandas as pd

def get_following_actions(df, col="Acceleration"):
    # Get the action that follows each acceleration
    mask = df['High Intensity Activity Type'] == f'{col}'
    following_actions = df.loc[mask].index + 1
    
    # Filter valid indices and get their actions
    valid_actions = df.loc[following_actions[following_actions < len(df)]]
    
    # Count and sort
    return valid_actions['High Intensity Activity Type'].value_counts()

# Usage
following_actions = get_following_actions(df, 'Deceleration')
print(following_actions)

# %% 
#Train multiple action skill
# @skill
# def multiple_actions_in_a_timespan(df, action, time_in_seconds):
#     """
#     Identifies occurrences of multiple actions within a specified timespan.
    
#     Parameters:
#     df (pandas.DataFrame): DataFrame containing player actions
#     action (str): The type of action to analyze (e.g., 'Acceleration')
#     time_in_seconds (int): The time window to check for multiple actions
    
#     Returns:
#     dict: Dictionary containing analysis results
#     """
#     # Convert Start Time to datetime if not already
#     df['Start Time'] = pd.to_datetime(df['Start Time'])
    
#     # Filter for specific action
#     action_df = df[df['High Intensity Activity Type'] == action].copy()
#     action_df = action_df.sort_values('Start Time')
    
#     # Initialize variables for tracking windows
#     windows = []
#     current_window = []
    
#     # Analyze each action
#     for idx, row in action_df.iterrows():
#         current_time = row['Start Time']
        
#         # If window is empty, add first action
#         if not current_window:
#             current_window = [(current_time, row)]
#             continue
        
#         # Check if current action is within timespan of window start
#         time_diff = (current_time - current_window[0][0]).total_seconds()
        
#         if time_diff <= time_in_seconds:
#             # Add to current window if within timespan
#             current_window.append((current_time, row))
#         else:
#             # If current window has multiple actions, save it
#             if len(current_window) > 1:
#                 windows.append(current_window)
#             # Start new window with current action
#             current_window = [(current_time, row)]
    
#     # Add final window if it has multiple actions
#     if len(current_window) > 1:
#         windows.append(current_window)
    
#     # Calculate statistics
#     total_windows = len(windows)
#     if total_windows > 0:
#         actions_per_window = [len(window) for window in windows]
#         avg_actions = sum(actions_per_window) / total_windows
#         max_actions = max(actions_per_window)
#     else:
#         avg_actions = 0
#         max_actions = 0
    
#     # return {
#     #     'total_windows': total_windows,
#     #     'avg_actions_per_window': round(avg_actions, 2),
#     #     'max_actions_in_window': max_actions,
#     #     'windows': windows
#     # }

#     return total_windows

# @skill
# def average_time_between_actions(df, action):
#     """
#     Calculates the average time between specified actions using 'Time Since Last' column.
    
#     Parameters:
#     df (pandas.DataFrame): DataFrame containing player actions
#     action (str): The type of action to analyze (e.g., 'Acceleration')
    
#     Returns:
#     dict: Dictionary containing average time between actions and additional statistics
#     """
#     # Filter for specific action
#     action_df = df[df['High Intensity Activity Type'] == action].copy()
    
#     # Convert 'Time Since Last' from string format (MM:SS.S) to seconds
#     time_diffs = action_df[action_df['Time Since Last'] != '00:00.0']['Time Since Last'].apply(
#         lambda x: float(x.split(':')[0]) * 60 + float(x.split(':')[1])
#     ).tolist()
    
#     if time_diffs:
#         average_time=round(sum(time_diffs) / len(time_diffs), 2)
#         return {
#             'average_time': round(sum(time_diffs) / len(time_diffs), 2),
#             'min_time': round(min(time_diffs), 2),
#             'max_time': round(max(time_diffs), 2),
#             'total_actions': len(action_df),
#             'time_differences': time_diffs
#         }
#     else:
#         # return {
#         #     'average_time': 0,
#         #     'min_time': 0,
#         #     'max_time': 0,
#         #     'total_actions': len(action_df),
#         #     'time_differences': []
#         # }
#         return average_time

# @skill
# def analyze_preceding_actions(df, target_action='Acceleration'):
#     """
#     Analyzes what actions most frequently precede a specified HIA type.
    
#     Parameters:
#     df (pandas.DataFrame): DataFrame containing player actions
#     target_action (str): The HIA type to analyze (default: 'Acceleration')
    
#     Returns:
#     tuple: (detailed_results, (most_common_action, percentage))
#     """
#     # Filter for the target action
#     target_rows = df[df['High Intensity Activity Type'] == target_action]
    
#     # Count preceding actions
#     preceding_counts = target_rows['Preceding High Intensity Activity Type'].value_counts()
    
#     # Calculate percentages
#     total_occurrences = len(target_rows)
#     preceding_percentages = (preceding_counts / total_occurrences * 100).round(2)
    
#     # Get most common action and its percentage
#     most_common_action = preceding_counts.index[0]
#     most_common_percentage = preceding_percentages[most_common_action]
    
#     # Detailed results
#     results = {
#         'total_occurrences': total_occurrences,
#         'preceding_actions': {
#             action: {
#                 'count': count,
#                 'percentage': preceding_percentages[action]
#             }
#             for action, count in preceding_counts.items()
#         }
#     }
    
#     return (most_common_action, most_common_percentage)
@skill
def multiple_actions_in_a_timespan(df, action, time_in_seconds):
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

@skill
def average_time_between_actions(df, action):
    """
    Calculates the average time between specified actions using 'Time Since Last' column.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing player actions
    action (str): The type of action to analyze (e.g., 'Acceleration')
    
    Returns:
    dict: Formatted response with average time between actions
    """
    # Filter for specific action
    action_df = df[df['High Intensity Activity Type'] == action].copy()
    
    # Convert 'Time Since Last' from string format (MM:SS.S) to seconds
    time_diffs = action_df[action_df['Time Since Last'] != '00:00.0']['Time Since Last'].apply(
        lambda x: float(x.split(':')[0]) * 60 + float(x.split(':')[1])
    ).tolist()
    
    if time_diffs:
        avg_time = round(sum(time_diffs) / len(time_diffs), 2)
        return {
            'type': 'number',
            'value': float(avg_time)
        }
    return {
        'type': 'number',
        'value': 0.0
    }

@skill
def analyze_preceding_actions(df, target_action='Acceleration'):
    """
    Analyzes what actions most frequently precede a specified HIA type.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing player actions
    target_action (str): The HIA type to analyze (default: 'Acceleration')
    
    Returns:
    dict: Formatted response with most common preceding action and its percentage
    """
    # Filter for the target action
    target_rows = df[df['High Intensity Activity Type'] == target_action]
    
    # Count preceding actions
    preceding_counts = target_rows['Preceding High Intensity Activity Type'].value_counts()
    
    # Calculate percentages
    total_occurrences = len(target_rows)
    preceding_percentages = (preceding_counts / total_occurrences * 100).round(2)
    
    # Get most common action and its percentage
    most_common_action = preceding_counts.index[0]
    most_common_percentage = preceding_percentages[most_common_action]
    
    return {
        'type': 'string',
        'value': f"{most_common_action} ({most_common_percentage}%)"
    }

# %%
from pandasai import Agent
from pandasai.ee.vectorstores import ChromaDB



# # Instantiate the vector store
vector_store = ChromaDB()
agent = Agent(df, vectorstore=vector_store, config={
        "field_descriptions": field_descriptions,
        "verbose": True,
        'llm': llm})  # Helpful for debugging)

# # Better approach for skills registration
# #agent = Agent(data, vectorstore=vector_store)
# agent = Agent(
#     data, 
#     vectorstore=vector_store,
#     config={
#         "field_descriptions": field_descriptions,
#         "verbose": True  # Helpful for debugging
#     }
# )
# agent.add_skills([
#     multiple_actions_in_a_timespan,
#     average_time_between_actions,
#     analyze_preceding_actions
# ])
query = "What action most often follows an acceleration?"
query1 = "What is the action that most often comes before a decelration"
query2 = "How many long sprints were completed?"
query3 = "How often do players complete multiple sprints in 1 minute"
query4 = "How often do players complete multiple accelerations in 30 seconds"
query5 = "What is the average time between accelerations"

response = '''
import pandas as pd

def get_following_actions(df):
    # Get the action that follows each acceleration
    mask = df['High Intensity Activity Type'] == 'Acceleration'
    following_actions = df.loc[mask].index + 1
    
    # Filter valid indices and get their actions
    valid_actions = df.loc[following_actions[following_actions < len(df)]]
    
    # Count and sort
    return valid_actions['High Intensity Activity Type'].value_counts()

# Usage
following_actions = get_following_actions(df)
following_actions
'''

response1 = '''
import pandas as pd
(most_common, percentage) = analyze_preceding_actions(df, 'Acceleration')

'''
response2 = '''
df['Long_sprint'].sum()
'''

response3 = '''
import pandas as pd
result = multiple_actions_in_a_timespan(df, 'Sprint', 60)
'''

response4 = '''
import pandas as pd
result = multiple_actions_in_a_timespan(df, 'Acceleration', 30)
'''

response5 = '''
import pandas as pd
result = average_time_between_actions(david_df, 'Acceleration')
'''


# # Define training data with proper formatting
# training_data = [
#     {
#         "query": "What action most often follows an acceleration?",
#         "code": """
# def get_following_actions(df):
#     # Get the action that follows each acceleration
#     mask = df['High Intensity Activity Type'] == 'Acceleration'
#     following_actions = df.loc[mask].index + 1
    
#     # Filter valid indices and get their actions
#     valid_actions = df.loc[following_actions[following_actions < len(df)]]
    
#     # Count and sort
#     counts = valid_actions['High Intensity Activity Type'].value_counts()
#     return {
#         'type': 'string',
#         'value': str(counts)
#     }

# result = get_following_actions(df)
# result
# """
#     },
#     {
#         "query": "What action most often comes before a deceleration?",
#         "code": """
# result = analyze_preceding_actions(df, 'Deceleration')
# result
# """
#     },
#     {
#         "query": "How many long sprints were completed?",
#         "code": """
# result = {
#     'type': 'number',
#     'value': float(df['Long_sprint'].sum())
# }
# result
# """
#     },
#     {
#         "query": "How often do players complete multiple sprints in 1 minute?",
#         "code": """
# result = multiple_actions_in_a_timespan(df, 'Sprint', 60)
# result
# """
#     },
#     {
#         "query": "How often do players complete multiple accelerations in 30 seconds?",
#         "code": """
# result = multiple_actions_in_a_timespan(df, 'Acceleration', 30)
# result
# """
#     },
#     {
#         "query": "What is the average time between accelerations?",
#         "code": """
# result = average_time_between_actions(df, 'Acceleration')
# result
# """
#     }
# ]

# # Setup and train the agent
# def setup_trained_agent(data):
#     """
#     Sets up and trains the PandasAI agent with the defined skills and training data.
    
#     Parameters:
#     data (pandas.DataFrame): The DataFrame to analyze
    
#     Returns:
#     Agent: Trained PandasAI agent

#     """
#     # Instantiate the vector store
#     llm = ChatGroq(
#         model="llama3-8b-8192",
#         temperature=0,
#         max_tokens=None,
#         timeout=None,
#         max_retries=2,
#     )
#     vector_store = ChromaDB()
    
#     # Create the agent
#     agent = Agent(data, vectorstore=vector_store, config = {'llm': llm})
    
#     # Add skills
#     agent.add_skills([
#         multiple_actions_in_a_timespan,
#         average_time_between_actions,
#         analyze_preceding_actions
#     ])
    
#     # Train the agent
#     agent.train(
#         queries=[item["query"] for item in training_data],
#         codes=[item["code"] for item in training_data]
#     )
    
#     return agent

# # Example usage:
# # data = pd.read_csv("your_data.csv")
# agent = setup_trained_agent(data)
# %%
print("---------------------What action most often follows a deceleration?-----------------------")
agent.train(queries=[query,query1,query2,query3,query4,query5], codes=[response,response1,response2,response3,response4,response5,])
response = agent.chat("What action most often follows a deceleration?")
print(response)
print("--------------------------------------------")
# %%
print("-------------------How many long sprints were completed?-------------------------")
response = agent.chat("How many long sprints were completed?")
print(response)
print("--------------------------------------------")
# %%
print("--------------------How often are multiple decelerations completed in 20 seconds?------------------------")
response = agent.chat("How often are multiple decelerations completed in 40 seconds?")
print(response)
print("--------------------------------------------")
# %%
