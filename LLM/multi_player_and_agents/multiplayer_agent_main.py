from typing import List, Dict, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict
import json
from crewai.tools import tool
from crewai.tools import BaseTool
class SplitDataframeInput(BaseModel):
    players: Optional[List[str]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
from llm import get_llm  
from crewai import Agent, Task, Crew, Process

llm = get_llm()
from typing import List, Dict, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict
import json
from crewai.tools import tool
from crewai.tools import BaseTool

# First, initialise our global DataFrame manager
class DataFrameManager:
    _instance = None
    _df = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DataFrameManager()
        return cls._instance

    def set_dataframe(self, df):
        self._df = df

    def get_dataframe(self):
        return self._df

# Modified Pydantic model with arbitrary types allowed
class SplitDataframeInput(BaseModel):
    players: Optional[List[str]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)  # This is crucial

# Initialize the DataFrame manager
df_manager = DataFrameManager.get_instance()
df_manager.set_dataframe(pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-12-16 Players-HIAs-Export.csv"))

@tool("Split dataframe by players")
def split_dataframe(input_data: Union[str, dict, SplitDataframeInput]) -> Dict[str, pd.DataFrame]:
    """
    Splits the main dataframe into separate dataframes for specified players.
    """
    try:
        df = df_manager.get_dataframe()
        if df is None:
            print("DataFrame is None - check DataFrame manager")
            return {}

        # Parse input data
        if isinstance(input_data, str):
            try:
                parsed_data = json.loads(input_data)
                players = parsed_data.get("players", [])
            except json.JSONDecodeError:
                print("Invalid JSON string")
                return {}
        elif isinstance(input_data, dict):
            players = input_data.get("players", [])
        elif isinstance(input_data, SplitDataframeInput):
            players = input_data.players or []
        else:
            print(f"Unexpected input type: {type(input_data)}")
            return {}

        # Debug information about DataFrame structure
        print("Processing request for players:", players)
        print("DataFrame columns:", df.columns.tolist())

        # Check for player column
        player_column = None
        potential_player_columns = ['Player', 'player', 'Name', 'name', 'PlayerName', 'Player Name']
        for col in potential_player_columns:
            if col in df.columns:
                player_column = col
                break

        if player_column is None:
            print("Could not find player column. Available columns are:", df.columns.tolist())
            return {}

        # Filter dataframes
        player_dfs = {}
        for player in players:
            if player in df[player_column].unique():
                player_dfs[player] = df[df[player_column] == player].copy()
                print(f"Found data for {player}: {len(player_dfs[player])} rows")
            else:
                print(f"No data found for player: {player}")

        return player_dfs

    except Exception as e:
        print(f"Error in split_dataframe: {str(e)}")
        return {}
# Rest of your agent and crew setup remains the same
dataframe_splitter = Agent(
    llm=llm,
    role="Split Dataframe",
    goal="Split the dataframe to create separate dataframes for the input players",
    backstory="""I am a data specialist who filters dataframes by player names. 
    I take the main dataframe and return individual dataframes for specified players.""",
    tools=[split_dataframe],
    allow_code_execution=True,
    code_execution_mode="unsafe",
    verbose=True,
    use_system_prompt=False
)

# Add this function to parse player names from the prompt
def extract_players_from_prompt(prompt: str) -> List[str]:
    """Extract player names from user prompt."""
    try:
        # Simple parsing: Assumes format "Filter the dataframe to X and Y"
        if "Filter the dataframe to" in prompt:
            players_part = prompt.split("Filter the dataframe to")[1].split(".")[0]
            players = [p.strip() for p in players_part.split("and")]
            return players
        return []
    except Exception as e:
        print(f"Error parsing prompt: {str(e)}")
        return []

# Modify the Task to accept and use input parameters
identify_players = Task(
    description="""
    Using the split_dataframe tool:
    1. Filter the main dataframe to only include data for the specified players in the prompt: {players}
    2. Return a dictionary with their individual dataframes
    The input should be: {{"players": {players}}}
    """,
    expected_output="A dictionary containing the filtered dataframes for the specified players",
    agent=dataframe_splitter#,
  #  ,context={"data_available": True}  # Add any context needed
)
crew = Crew(
    agents=[dataframe_splitter],
    tasks=[identify_players],
    process=Process.sequential
)
# Modify how you execute the crew
user_prompt = 'Filter the dataframe to Mark and Lee. This is the relevant dataframe'#: \n {df}'
players = extract_players_from_prompt(user_prompt)

try:
    result = crew.kickoff(
        inputs={
            "players": players,
            "user_prompt": user_prompt
        }
    )
    print("Task completed successfully")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error details: {str(e)}")