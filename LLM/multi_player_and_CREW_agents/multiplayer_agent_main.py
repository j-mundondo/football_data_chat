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
        df['Player'] = df['Player'].str.strip().str.lower()
        self._df = df[(df['Player'] == 'kolo') | (df['Player'] == 'lee') | (df['Player'] == 'mark')]


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
#df_manager.set_dataframe(pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-12-16 Players-HIAs-Export.csv"))
df_manager.set_dataframe(pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-15-17 Players-HIAs-Export_.csv"))

@tool("Split dataframe by players")
def split_dataframe(input_data: Union[str, dict, SplitDataframeInput]) -> Dict[str, pd.DataFrame]:
    """
    Filters the main dataframe to only include the specified players.
    Returns a dictionary with a single key containing the filtered dataframe.
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
        player_column = 'Player'

        if player_column is None:
            print("Could not find player column. Available columns are:", df.columns.tolist())
            return {}

        # Debug: Print unique players in DataFrame
        available_players = df[player_column].unique()
        print(f"Available players in DataFrame: {available_players}")

        # Filter dataframe for all specified players at once, with case-insensitive matching
        mask = df[player_column].str.lower().isin([p.lower() for p in players])
        filtered_df = df[mask].copy()
        
        # Print debug information
        if len(filtered_df) == 0:
            print("No data found for any specified players")
            return {}
            
        print(f"Total rows in filtered DataFrame: {len(filtered_df)}")
        # Group by player to show count per player
        player_counts = filtered_df.groupby(player_column).size()
        print("Rows per player:")
        print(player_counts)

        return {"filtered_data": filtered_df}

    except Exception as e:
        print(f"Error in split_dataframe: {str(e)}")
        print(f"Error details: {str(e.__class__.__name__)}")
        return {}

@tool("Compare players statistics")
def compare_players(input_data: Union[str, dict, ComparePlayersInput]) -> Dict:
    """
    Compares statistics between players based on metrics from a single filtered dataframe.
    
    Example:
        input_data = {
            "filtered_data": dataframe_with_both_players,
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
        
        # Extract the filtered DataFrame
        if not isinstance(input_data, dict) or 'filtered_data' not in input_data:
            return {"error": "Missing filtered_data in input"}
            
        df = input_data['filtered_data']
        
        # Convert to DataFrame if it's a dict
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        
        # Get unique players from the filtered DataFrame
        players = df['Player'].lower().unique()
        if len(players) != 2:
            return {"error": f"Expected 2 players, found {len(players)}"}
            
        player_one, player_two = players
            
        # Create separate views for each player
        player_dfs = {
            player_one: df[df['Player'] == player_one],
            player_two: df[df['Player'] == player_two]
        }
            
        # Apply action type filter if specified
        if 'action_type' in input_data and input_data['action_type'] and input_data['action_type'].lower() != "none":
            for player in players:
                player_dfs[player] = player_dfs[player][
                    player_dfs[player]['High Intensity Activity Type'] == input_data['action_type']
                ]
        
        # Get metrics (either from input or default)
        metrics = input_data.get('metrics', [
            'Distance', 'Magnitude', 'Avg Metabolic Power', 'Dynamic Stress Load'
        ])
        
        # Calculate comparisons
        comparisons = {}
        for metric in metrics:
            if metric not in df.columns:
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
                "players": list(players),
                "metrics_compared": metrics,
                "action_type": input_data.get('action_type', "all actions")
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
    ,max_retry_limit=1
    , 
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
    ,max_retry_limit=1
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
    ,output_pydantic=SplitDataframeInput
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
    ,output_pydantic=ComparePlayersInput
    ,max_retries=2
)

# Setup and execute the crew
crew = Crew(
    agents=[dataframe_splitter, player_comparison_agent],
    tasks=[identify_players, compare_the_players],
    process=Process.sequential,
    max_retries=2,
    verbose=True
    
)
#print(df_manager.get_dataframe()['Player'].value_counts())
#Execute with proper inputs
try:
    # user_prompt = 'Filter the dataframe to Colly and Lee. This is the relevant dataframe'
    # players = extract_players_from_prompt(user_prompt)
    
    players = ['kolo', 'lee']
    user_prompt = f'Filter the dataframe to {players[0]} and {players[1]} and then subsequently compare their statistics.'

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


# # if result:
# #     for i in result:
# #         print(f"--------- \n {type(i)} \n -----------------")