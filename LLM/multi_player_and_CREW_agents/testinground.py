from typing import List, Dict, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict
import json
from crewai.tools import tool
from llm import get_llm, get_llm_2
from crewai import Agent, Task, Crew, Process

# Get LLMs
llm, llm2 = get_llm(), get_llm_2()

class DataFrameManager:
    _instance = None
    _df = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DataFrameManager()
        return cls._instance

    def set_dataframe(self, df: pd.DataFrame):
        """Set the dataframe with validation."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if 'Player' not in df.columns:
            raise ValueError("DataFrame must contain 'Player' column")
            
        # Clean and standardize player names
        df['Player'] = df['Player'].str.strip().str.lower()
        
        # Store the DataFrame
        self._df = df.copy()
        print(f"DataFrame set with {len(self._df)} rows and {len(self._df.columns)} columns")
        print(f"Available players: {self._df['Player'].unique()}")

    def get_dataframe(self) -> pd.DataFrame:
        """Get the dataframe with validation."""
        if self._df is None:
            raise ValueError("DataFrame not initialized. Call set_dataframe first.")
        return self._df.copy()

    def validate_players(self, players: List[str]) -> bool:
        """Validate that requested players exist in the dataframe."""
        if self._df is None:
            return False
        available_players = set(self._df['Player'].unique())
        requested_players = set([p.lower().strip() for p in players])
        return requested_players.issubset(available_players)

class SplitDataframeInput(BaseModel):
    players: Optional[List[str]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ComparePlayersInput(BaseModel):
    filtered_data: pd.DataFrame
    metrics: Optional[List[str]] = None
    action_type: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

@tool("Split dataframe by players")
def split_dataframe(input_data: Union[str, dict, SplitDataframeInput]) -> Dict[str, pd.DataFrame]:
    """Filters the main dataframe to only include the specified players."""
    try:
        df_manager = DataFrameManager.get_instance()
        df = df_manager.get_dataframe()
        
        # Parse input data
        if isinstance(input_data, str):
            try:
                parsed_data = json.loads(input_data)
                players = parsed_data.get("players", [])
            except json.JSONDecodeError:
                return {"error": "Invalid JSON string"}
        elif isinstance(input_data, dict):
            players = input_data.get("players", [])
        elif isinstance(input_data, SplitDataframeInput):
            players = input_data.players or []
        else:
            return {"error": f"Unexpected input type: {type(input_data)}"}

        # Validate players
        if not df_manager.validate_players(players):
            return {"error": "One or more players not found in dataset"}

        # Filter dataframe for specified players
        players = [p.lower().strip() for p in players]
        filtered_df = df[df['Player'].isin(players)].copy()
        
        print(f"Filtered DataFrame contains {len(filtered_df)} rows")
        print(f"Players in filtered data: {filtered_df['Player'].unique()}")
        
        return {"filtered_data": filtered_df}

    except Exception as e:
        print(f"Error in split_dataframe: {str(e)}")
        return {"error": str(e)}

@tool("Compare players statistics")
def compare_players(input_data: Union[str, dict, ComparePlayersInput]) -> Dict:
    """Compares statistics between players."""
    try:
        # Handle input parsing
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON string"}

        # Get the filtered DataFrame
        if isinstance(input_data, dict):
            if 'filtered_data' not in input_data:
                return {"error": "Missing filtered_data"}
            df = input_data['filtered_data']
        else:
            df = input_data.filtered_data

        # Convert to DataFrame if needed
        if isinstance(df, dict):
            df = pd.DataFrame(df)

        # Verify we have the required columns
        required_metrics = ['Distance', 'Magnitude', 'Avg Metabolic Power', 'Dynamic Stress Load']
        missing_metrics = [metric for metric in required_metrics if metric not in df.columns]
        if missing_metrics:
            return {"error": f"Missing required metrics: {missing_metrics}"}

        # Group by player and calculate statistics
        player_stats = {}
        for player in df['Player'].unique():
            player_df = df[df['Player'] == player]
            stats = {}
            for metric in required_metrics:
                metric_stats = player_df[metric].describe()
                stats[metric] = {
                    'mean': float(metric_stats['mean']),
                    'max': float(metric_stats['max']),
                    'min': float(metric_stats['min']),
                    'total': float(player_df[metric].sum()),
                    'count': int(metric_stats['count'])
                }
            player_stats[player] = stats

        # Calculate comparisons between players
        players = list(player_stats.keys())
        comparisons = {}
        if len(players) == 2:
            for metric in required_metrics:
                player1, player2 = players
                comparisons[metric] = {
                    'difference_mean': player_stats[player1][metric]['mean'] - player_stats[player2][metric]['mean'],
                    'difference_total': player_stats[player1][metric]['total'] - player_stats[player2][metric]['total']
                }

        return {
            'player_stats': player_stats,
            'comparisons': comparisons,
            'metrics_compared': required_metrics
        }

    except Exception as e:
        print(f"Error in compare_players: {str(e)}")
        return {"error": str(e)}

# Define agents
dataframe_splitter = Agent(
    role="Split Dataframe",
    goal="Split the dataframe to create separate dataframes for the input players",
    backstory="I am a data specialist who filters dataframes by player names.",
    tools=[split_dataframe],
    llm=llm,
    verbose=True,
    allow_delegation=False,
    allow_code_execution=True,
    code_execution_mode="unsafe",
    use_system_prompt=False,
    max_retry_limit=1
)

player_comparison_agent = Agent(
    role="Statistics Analyst",
    goal="Compare player metrics and provide detailed statistical analysis",
    backstory="I am a statistical analyst specialising in player performance metrics.",
    tools=[compare_players],
    llm=llm2,
    verbose=True,
    allow_delegation=False,
    allow_code_execution=True,
    code_execution_mode="unsafe",
    use_system_prompt=False,
    max_retry_limit=1
)

# Define tasks
identify_players = Task(
    description="""
    Using the split_dataframe tool:
    1. Filter the dataframe to include ONLY these players: {players}
    2. The data should include all columns: Session, Date, Drill, Player, HIA Type, etc.
    3. Return the complete filtered dataframe as a dictionary
    """,
    expected_output="A dictionary containing the filtered dataframe with all player data",
    agent=dataframe_splitter,
    max_retries=2
)

compare_players_task = Task(
    description="""
    Using the compare_players tool:
    1. Take the filtered dataframe from the previous task
    2. Compare these specific players: {players}
    3. Calculate statistics for each metric:
       - Distance
       - Magnitude
       - Avg Metabolic Power
       - Dynamic Stress Load
    4. Return detailed statistics including means, totals, and counts
    """,
    expected_output="Complete statistical comparison between players for all metrics",
    agent=player_comparison_agent,
    context=[identify_players],
    dependencies=[identify_players],
    max_retries=2
)

# Setup crew
crew = Crew(
    agents=[dataframe_splitter, player_comparison_agent],
    tasks=[identify_players, compare_players_task],
    process=Process.sequential,
    verbose=True,
    max_retries=2
)

def run_analysis(players: List[str], data_path: str):
    """Run the complete analysis workflow."""
    try:
        # Initialize DataFrame
        df_manager = DataFrameManager.get_instance()
        df = pd.read_csv(data_path)
        df_manager.set_dataframe(df)
        
        # Verify DataFrame initialization
        test_df = df_manager.get_dataframe()
        if test_df is None or len(test_df) == 0:
            raise ValueError("DataFrame not properly initialized")
        
        # Run the analysis
        result = crew.kickoff(inputs={
            "players": players
        })
        
        return result
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    data_path = r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-15-17 Players-HIAs-Export_.csv"  # Update with your actual path
    players = ['kolo', 'lee']
    
    print("Starting analysis...")
    result = run_analysis(players, data_path)
    
    if result:
        print("\nAnalysis Results:")
        print(json.dumps(result, indent=2))
    else:
        print("\nAnalysis failed")