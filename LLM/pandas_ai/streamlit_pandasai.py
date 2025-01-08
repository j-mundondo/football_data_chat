from pathlib import Path
import streamlit as st
from pandasai.responses.streamlit_response import StreamlitResponse
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
import tempfile
#from streamlit_chromadb_connection.chromadb_connection import ChromadbConnection

# Load environment variables
load_dotenv()

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

# Create a temporary directory for cache
@st.cache_resource
def init_cache():
    temp_dir = tempfile.mkdtemp()
    os.environ['PANDASAI_CACHE_DIR'] = temp_dir
    return temp_dir

# Initialize cache directory
cache_dir = init_cache()

# SET UP LLM
@st.cache_resource
def init_llm():
    return ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

# Initialize LLM
llm = init_llm()

# Title of the web app
st.title('Football Movement Analysis')

# Upload a file
uploaded_file = st.file_uploader("Choose a file")

# If a file is uploaded
if uploaded_file is not None:
    # Read the file as a DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Initialize agent with the data
        @st.cache_resource
        def init_agent(_df):
          #  vector_store = ChromaDB()
            return Agent(
                _df,
               # vectorstore=vector_store,
                memory_size=2,
                config={
                    "field_descriptions": field_descriptions,
                    'llm': llm,
                    "verbose": True,
                    "custom_whitelisted_dependencies": ["dateutil", "matplotlib", "seaborn"]
                }
            )
        
        # Create agent instance
        agent = init_agent(df)
        
        # Add user input
        user_question = st.text_input("Ask a question about your data:")
        if user_question:
            with st.spinner('Analyzing...'):
                response = agent.chat(user_question)
                st.write("Response:", response)
                
                # Show generated code in expander
                with st.expander("See generated code"):
                    st.code(agent.last_code_generated)
                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")