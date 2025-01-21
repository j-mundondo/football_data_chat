from typing import List, Dict, Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict
import json
from crewai.tools import tool
from crewai.tools import BaseTool

def multi_agent_tools(df: pd.DataFrame):
    """Creates and returns tools for multi-agent analysis of player data."""
    
    class SplitDataframeInput(BaseModel):
        players: Optional[List[str]] = None
        model_config = ConfigDict(arbitrary_types_allowed=True)
    
    class ComparePlayersInput(BaseModel):
        player_one: str
        player_two: str
        metrics: Optional[List[str]] = None
        action_type: Optional[str] = None
        model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @tool("Split dataframe by players")
    def split_dataframe(df,input_data: Union[str, dict, SplitDataframeInput]) -> Dict[str, pd.DataFrame]:
        """
        Splits the main dataframe into separate dataframes for specified players.
        
        Args:
            input_data: Can be:
                - String (JSON formatted): {"players": ["player1", "player2"]}
                - Dictionary with players key
                - SplitDataframeInput instance
            
        Returns:
            Dictionary with player names as keys and their respective dataframes as values
        
        Example:
            input_data = {"players": ["Mark", "Kolo"]}
            returns {'Mark': df_mark, 'Kolo': df_kolo}
        """
        # Handle different input types
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON string provided"}
                
        if isinstance(input_data, dict):
            input_data = SplitDataframeInput(**input_data)
            
        players = input_data.players if input_data.players else df['Player'].unique().tolist()
        
        player_dfs = {}
        for player in players:
            if player in df['Player'].unique():
                player_dfs[player] = df[df['Player'] == player].copy()
                
        return player_dfs

    @tool("Compare players statistics")
    def compare_players(df,input_data: Union[str, dict, ComparePlayersInput]) -> Dict:
        """
        Compares statistics between two players based on specified metrics.
        
        Args:
            input_data: Can be:
                - String (JSON formatted): {"player_one": "Mark", "player_two": "Kolo", "metrics": ["Distance"]}
                - Dictionary with player_one, player_two, metrics keys
                - ComparePlayersInput instance
            
        Returns:
            Dictionary containing comparative statistics including:
                - Total counts
                - Means
                - Maximums
                - Minimums
                - Differences
        
        Example:
            input_data = {
                "player_one": "Mark",
                "player_two": "Kolo",
                "metrics": ["Distance"],
                "action_type": "Sprint"
            }
        """
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
                    # If it's a single metric in string form, convert to list
                    if input_data['metrics'].strip('[]\'\"'):
                        input_data['metrics'] = [input_data['metrics'].strip('[]\'\"')]
                        
            input_data = ComparePlayersInput(**input_data)
            
        # Get player dataframes using the split_dataframe function
        split_input = SplitDataframeInput(players=[input_data.player_one, input_data.player_two])
        player_dfs = split_dataframe(split_input)
        
        if not all(player in player_dfs for player in [input_data.player_one, input_data.player_two]):
            return {"error": "One or more players not found in dataset"}
            
        # Apply action type filter if specified
        if input_data.action_type and input_data.action_type.lower() != "none":
            for player in player_dfs:
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
            if metric not in player_dfs[input_data.player_one].columns:
                continue
                
            stats_1 = player_dfs[input_data.player_one][metric].describe()
            stats_2 = player_dfs[input_data.player_two][metric].describe()
            
            comparisons[metric] = {
                f"{input_data.player_one}_stats": {
                    "mean": float(stats_1['mean']),
                    "max": float(stats_1['max']),
                    "min": float(stats_1['min']),
                    "total": float(player_dfs[input_data.player_one][metric].sum()),
                    "count": int(stats_1['count'])
                },
                f"{input_data.player_two}_stats": {
                    "mean": float(stats_2['mean']),
                    "max": float(stats_2['max']),
                    "min": float(stats_2['min']),
                    "total": float(player_dfs[input_data.player_two][metric].sum()),
                    "count": int(stats_2['count'])
                },
                "difference": {
                    "mean": float(stats_1['mean'] - stats_2['mean']),
                    "total": float(player_dfs[input_data.player_one][metric].sum() - 
                                 player_dfs[input_data.player_two][metric].sum())
                }
            }
        
        return {
            "comparison_details": {
                "players": [input_data.player_one, input_data.player_two],
                "metrics_compared": metrics,
                "action_type": input_data.action_type if input_data.action_type else "all actions"
            },
            "comparisons": comparisons
        }

    def get_all_tools() -> List:
        """Returns list of all available tools for the agents"""
        return [split_dataframe, compare_players]
        
    return get_all_tools()
