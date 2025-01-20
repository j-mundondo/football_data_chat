from typing import List, Dict, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict
import json
from crewai.tools import tool
from llm import get_llm , get_llm_2
from crewai import Agent, Task, Crew, Process
from typing import List, Dict, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict
import json
from crewai.tools import tool


llm, llm2 = get_llm(),get_llm_2()

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

class ComparePlayersInput(BaseModel):
    player_dfs: Dict[str, pd.DataFrame]
    metrics: Optional[List[str]] = None
    action_type: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


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

@tool("Compare players statistics")
def compare_players(input_data: Union[str, dict, ComparePlayersInput]) -> Dict:
    """
    Compares statistics between two players based on specified metrics.
    
    Args:
        input_data: Can be:
            - String (JSON formatted): {
                "player_dfs": {
                    "Mark": mark_dataframe,
                    "Kolo": kolo_dataframe
                },
                "metrics": ["Distance"],
                "action_type": "Sprint"
            }
            - Dictionary with player_dfs, metrics keys
            - ComparePlayersInput instance
        
    Example:
        input_data = {
            "player_dfs": result_from_split_dataframe,
            "metrics": ["Distance"],
            "action_type": "Sprint"
        }
    """
    try:
        # Handle different input types
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON string provided"}
                
        if isinstance(input_data, dict):
            # Handle metrics if it's a string representation of a list
            if 'metrics' in input_data and isinstance(input_data['metrics'], str):
                try:
                    input_data['metrics'] = json.loads(input_data['metrics'])
                except json.JSONDecodeError:
                    if input_data['metrics'].strip('[]\'\"'):
                        input_data['metrics'] = [input_data['metrics'].strip('[]\'\"')]
                        
            input_data = ComparePlayersInput(**input_data)
        
        player_dfs = input_data.player_dfs
        players = list(player_dfs.keys())
        
        if len(players) != 2:
            return {"error": "Exactly two players are required for comparison"}

        player_one, player_two = players
            
        # Apply action type filter if specified
        if input_data.action_type and input_data.action_type.lower() != "none":
            for player in players:
                player_dfs[player] = player_dfs[player][
                    player_dfs[player]['High Intensity Activity Type'] == input_data.action_type
                ]
        
        # Default metrics if none specified
        metrics = input_data.metrics if input_data.metrics else [
            'Distance', 'Magnitude', 'Avg Metabolic Power', 'Dynamic Stress Load'
        ]
        
        # Calculate comparisons
        comparisons = {}
        for metric in metrics:
            if metric not in player_dfs[player_one].columns:
                print(f"Metric {metric} not found in data")
                continue
                
            stats_1 = player_dfs[player_one][metric].describe()
            stats_2 = player_dfs[player_two][metric].describe()
            
            comparisons[metric] = {
                f"{player_one}_stats": {
                    "mean": float(stats_1['mean']),
                    "max": float(stats_1['max']),
                    "min": float(stats_1['min']),
                    "total": float(player_dfs[player_one][metric].sum()),
                    "count": int(stats_1['count'])
                },
                f"{player_two}_stats": {
                    "mean": float(stats_2['mean']),
                    "max": float(stats_2['max']),
                    "min": float(stats_2['min']),
                    "total": float(player_dfs[player_two][metric].sum()),
                    "count": int(stats_2['count'])
                },
                "difference": {
                    "mean": float(stats_1['mean'] - stats_2['mean']),
                    "total": float(player_dfs[player_one][metric].sum() - 
                                player_dfs[player_two][metric].sum())
                }
            }
        
        return {
            "comparison_details": {
                "players": players,
                "metrics_compared": metrics,
                "action_type": input_data.action_type if input_data.action_type else "all actions"
            },
            "comparisons": comparisons
        }
    except Exception as e:
        print(f"Error in compare_players: {str(e)}")
        return {"error": str(e)}

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
    ,max_iter=2
)

# Define the comparison agent
player_comparison_agent = Agent(
    llm=llm2,
    role="Statistics Analyst",
    goal="Compare player metrics and provide detailed statistical analysis",
    backstory="""I am a statistical analyst specialising in player performance metrics. 
    I take pre-filtered player data and perform detailed comparisons of their statistics.""",
    tools=[compare_players],
    allow_code_execution=True,
    code_execution_mode="unsafe",
    verbose=True,
    allow_delegation=False
    ,max_iter=2
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
    max_retries=2,
    agent=dataframe_splitter#,
  #  ,context={"data_available": True}  # Add any context needed
)

# Modify the comparison task description
compare_the_players = Task(
    description="""
    Using the output from the previous task, compare these players: {players}
    
    Compare all available metrics including:
    - Distance
    - Magnitude
    - Avg Metabolic Power
    - Dynamic Stress Load
    """,
    expected_output='Statistical comparison of relevant metrics between the two players',
    agent=player_comparison_agent,
    dependencies=[identify_players],
    context=[identify_players]
)

# Setup and execute the crew
crew = Crew(
    agents=[dataframe_splitter, player_comparison_agent],
    tasks=[identify_players, compare_the_players],
    process=Process.sequential,
    max_retries=2,
    verbose=True
)

# Execute with proper inputs
try:
    user_prompt = 'Filter the dataframe to Mark and Lee. This is the relevant dataframe'
    players = extract_players_from_prompt(user_prompt)
    
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


# if result:
#     for i in result:
#         print(f"--------- \n {type(i)} \n -----------------")