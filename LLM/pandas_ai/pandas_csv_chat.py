# %%
# BRING IN LIBRARIES
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import BambooLLM
from langchain_groq.chat_models import ChatGroq
from pandasai.connectors import PandasConnector
from dotenv import load_dotenv
load_dotenv()

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
    'Dynamic Stress Load': 'Cumulative stress metric from activity intensity'
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
data = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\individual_efforts\david_only_metrics.csv")
df = SmartDataframe(data,config = {'llm':llm})

# %%
df.chat('Find the average of the column named Dynamic Stress Load')


# %%
# bamboo_df = SmartDataframe(data,config = {'llm':bamboo_llm})
# response = bamboo_df.chat("Plot dynamic stress load on a line graph")
#print(response)

# %%
# DF WITH CONNECTOR

data = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\individual_efforts\david_only_metrics.csv")
connector = PandasConnector({"original_df": data}, field_descriptions=field_descriptions)
sdf = SmartDataframe(connector)
sdf.chat("What are the columns in the DF?")

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
following_actions = get_following_actions(data, 'Deceleration')
print(following_actions)


# %%
from pandasai import Agent
from pandasai.ee.vectorstores import ChromaDB
# Instantiate the vector store
vector_store = ChromaDB()
agent = Agent(data, vectorstore=vector_store)
query = "What action most often follows an acceleration?"

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

#agent.train(queries=[query], codes=[response])
response = agent.chat("What what action most often follows a deceleration?")
print(response)
# %%
response = agent.chat("What what action most often follows a deceleration?")
print(response)