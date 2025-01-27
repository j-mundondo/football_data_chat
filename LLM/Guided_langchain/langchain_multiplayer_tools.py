import pandas as pd
from langchain.tools import tool
from typing import List, Dict, Optional, Type, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

def filter_players(df: pd.DataFrame, player_list: List[str]) -> pd.DataFrame:
    """
    Filter a DataFrame to only include players whose names are in the provided list.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing a 'Player' column
        player_list (List[str]): List of player names to filter by
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only the specified players
        
    Raises:
        ValueError: If 'Player' column is not found in DataFrame
        ValueError: If player_list is empty
    """
    # Input validation
    if 'Player' not in df.columns:
        raise ValueError("DataFrame must contain a 'Player' column")
    
    if not player_list:
        raise ValueError("Player list cannot be empty")
    
    # Clean the player names in both the DataFrame and the input list
    df_clean = df.copy()
    df_clean['Player_Clean'] = df_clean['Player'].str.strip().str.lower()
    clean_player_list = [name.strip().lower() for name in player_list]
    
    # Filter the DataFrame
    filtered_df = df_clean[df_clean['Player_Clean'].isin(clean_player_list)]
    
    # Remove the temporary cleaning column and return
    return filtered_df.drop('Player_Clean', axis=1)

def compare_players(df: pd.DataFrame, player_list: List[str]) -> Dict:
    """
    Filter DataFrame for specified players and generate comparative statistics 
    for key performance metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame with player data
        player_list (List[str]): List of player names to analyse
        
    Returns:
        Dict: Comparative statistics for each player with the following metrics:
            - Activity type distribution
            - Average duration, distance, magnitude
            - Average metabolic power and dynamic stress load
            - Time-based metrics
            
    Raises:
        ValueError: If 'Player' column is not found in DataFrame
        ValueError: If player_list is empty
    """
    # Input validation
    if 'Player' not in df.columns:
        raise ValueError("DataFrame must contain a 'Player' column")
    
    if not player_list:
        raise ValueError("Player list cannot be empty")
    
    # Clean the player names in both the DataFrame and the input list
    df_clean = df.copy()
    df_clean['Player_Clean'] = df_clean['Player'].str.strip().str.lower()
    clean_player_list = [name.strip().lower() for name in player_list]
    
    # Filter the DataFrame
    filtered_df = df_clean[df_clean['Player_Clean'].isin(clean_player_list)]
    
    if filtered_df.empty:
        raise ValueError("No data found for the specified players")
    
    # Define numeric columns and categorical columns
    numeric_cols = ['Duration', 'Distance', 'Magnitude', 
                   'Avg Metabolic Power', 'Dynamic Stress Load']
    
    # Initialize results dictionary
    results = {}
    
    # Generate statistics for each player
    for player in clean_player_list:
        player_df = filtered_df[filtered_df['Player_Clean'] == player]
        
        if player_df.empty:
            results[player] = {
                'data_found': False,
                'message': f'No data found for player: {player}'
            }
            continue
            
        # Calculate activity type distribution
        activity_dist = player_df['High Intensity Activity Type'].value_counts()
        total_activities = len(player_df)
        
        # Calculate numeric statistics
        numeric_stats = {}
        for col in numeric_cols:
            if col in player_df.columns:
                stats = player_df[col].describe()
                numeric_stats[col] = {
                    'mean': float(stats['mean']),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'std': float(stats['std']),
                    'count': int(stats['count'])
                }
        
        # Parse and analyse time-based metrics
        def parse_time_to_seconds(time_str):
            if pd.isna(time_str):
                return None
            if isinstance(time_str, str):
                if ':' in time_str:
                    minutes, seconds = time_str.split(':')
                    return abs(float(minutes)) * 60 + float(seconds)
            return None
        
        time_gaps = player_df['Time Since Last Any Action'].apply(parse_time_to_seconds)
        time_gaps = time_gaps.dropna()
        
        if not time_gaps.empty:
            time_stats = {
                'avg_time_between_actions': f"{int(time_gaps.mean()//60):02d}:{(time_gaps.mean()%60):05.2f}",
                'min_time_between_actions': f"{int(time_gaps.min()//60):02d}:{(time_gaps.min()%60):05.2f}",
                'max_time_between_actions': f"{int(time_gaps.max()//60):02d}:{(time_gaps.max()%60):05.2f}"
            }
        else:
            time_stats = {
                'avg_time_between_actions': 'N/A',
                'min_time_between_actions': 'N/A',
                'max_time_between_actions': 'N/A'
            }
        
        # Compile results for this player
        results[player] = {
            'data_found': True,
            'total_activities': total_activities,
            'activity_distribution': {
                activity: {
                    'count': int(count),
                    'percentage': float(count/total_activities * 100)
                }
                for activity, count in activity_dist.items()
            },
            'numeric_metrics': numeric_stats,
            'time_metrics': time_stats
        }
    
    return {
        'player_comparisons': results,
        'total_players_analysed': len(results),
        'players_with_data': sum(1 for p in results.values() if p.get('data_found', False))
    }


class PlayerComparisonInput(BaseModel):
    """Input schema for comparing multiple player metrics."""
    players: List[str] = Field(
        ..., 
        description="List of player names to compare",
        min_items=2  # Ensure at least 2 players are provided
    )

class ComparePlayerMetrics(BaseTool):
    """Tool for comparing metrics between multiple players."""
    name: str = Field("compare_player_metrics", description="Name of the tool")
    description: str = Field(
        """Compare detailed statistics between multiple players including:
        - Activity type distribution
        - Average duration, distance, magnitude
        - Average metabolic power and dynamic stress load
        - Time-based metrics
        
        Input should be a list of player names to compare.
        """,
        description="Description of what the tool does"
    )
    args_schema: Type[BaseModel] = Field(PlayerComparisonInput, description="Schema for tool arguments")
    df: Any = Field(default=None, exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _run(self, players: List[str]) -> str:
        """Execute the comparison between multiple players."""
        try:
            # Use the existing compare_players function
            comparison_results = compare_players(
                self.df, 
                player_list=players
            )
            
            # Format the results into a readable string
            output_parts = [f"Comparison between {', '.join(players)}:"]
            
            for player_name, stats in comparison_results['player_comparisons'].items():
                if not stats['data_found']:
                    output_parts.append(f"\nNo data found for player: {player_name}")
                    continue
                
                output_parts.append(f"\n{player_name.title()} Statistics:")
                output_parts.append(f"Total Activities: {stats['total_activities']}")
                
                # Activity distribution
                output_parts.append("\nActivity Distribution:")
                for activity, details in stats['activity_distribution'].items():
                    output_parts.append(
                        f"- {activity}: {details['count']} "
                        f"({details['percentage']:.1f}%)"
                    )
                
                # Numeric metrics
                output_parts.append("\nPerformance Metrics:")
                for metric, values in stats['numeric_metrics'].items():
                    output_parts.append(
                        f"- {metric}:"
                        f"\n  Average: {values['mean']:.2f}"
                        f"\n  Min: {values['min']:.2f}"
                        f"\n  Max: {values['max']:.2f}"
                    )
                
                # Time metrics
                output_parts.append("\nTime Metrics:")
                for metric, value in stats['time_metrics'].items():
                    output_parts.append(f"- {metric}: {value}")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Error comparing players: {str(e)}"

    async def _arun(self, players: List[str]) -> str:
        """Async implementation of the tool."""
        return self._run(players)


######
from typing import List, Dict, Optional, Type, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from datetime import datetime

def compare_player_sessions(dataframes: Dict[str, pd.DataFrame], player_name: str) -> Dict:
    """
    Compare a player's performance metrics across multiple sessions.
    
    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary of DataFrames with session dates as keys
        player_name (str): Name of the player to analyse
        
    Returns:
        Dict: Comparative statistics across sessions
    """
    results = {
        'player_name': player_name,
        'sessions_analysed': len(dataframes),
        'session_comparisons': {}
    }
    
    # Clean the player name
    clean_player_name = player_name.strip().lower()
    
    for session_date, df in dataframes.items():
        # Clean the DataFrame player names
        df_clean = df.copy()
        df_clean['Player_Clean'] = df_clean['Player'].str.strip().str.lower()
        
        # Filter for the specific player
        player_df = df_clean[df_clean['Player_Clean'] == clean_player_name]
        
        if player_df.empty:
            results['session_comparisons'][session_date] = {
                'data_found': False,
                'message': f'No data found for player in session: {session_date}'
            }
            continue
            
        # Define metrics to analyse
        numeric_cols = ['Duration', 'Distance', 'Magnitude', 
                       'Avg Metabolic Power', 'Dynamic Stress Load']
        
        # Calculate activity type distribution
        activity_dist = player_df['High Intensity Activity Type'].value_counts()
        total_activities = len(player_df)
        
        # Calculate numeric statistics
        numeric_stats = {}
        for col in numeric_cols:
            if col in player_df.columns:
                stats = player_df[col].describe()
                numeric_stats[col] = {
                    'mean': float(stats['mean']),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'std': float(stats['std']),
                    'count': int(stats['count'])
                }
        
        # Parse and analyse time-based metrics
        def parse_time_to_seconds(time_str):
            if pd.isna(time_str):
                return None
            if isinstance(time_str, str):
                if ':' in time_str:
                    minutes, seconds = time_str.split(':')
                    return abs(float(minutes)) * 60 + float(seconds)
            return None
        
        time_gaps = player_df['Time Since Last Any Action'].apply(parse_time_to_seconds)
        time_gaps = time_gaps.dropna()
        
        if not time_gaps.empty:
            time_stats = {
                'avg_time_between_actions': f"{int(time_gaps.mean()//60):02d}:{(time_gaps.mean()%60):05.2f}",
                'min_time_between_actions': f"{int(time_gaps.min()//60):02d}:{(time_gaps.min()%60):05.2f}",
                'max_time_between_actions': f"{int(time_gaps.max()//60):02d}:{(time_gaps.max()%60):05.2f}"
            }
        else:
            time_stats = {
                'avg_time_between_actions': 'N/A',
                'min_time_between_actions': 'N/A',
                'max_time_between_actions': 'N/A'
            }
        
        # Store session results
        results['session_comparisons'][session_date] = {
            'data_found': True,
            'total_activities': total_activities,
            'activity_distribution': {
                activity: {
                    'count': int(count),
                    'percentage': float(count/total_activities * 100)
                }
                for activity, count in activity_dist.items()
            },
            'numeric_metrics': numeric_stats,
            'time_metrics': time_stats
        }
    
    # Add trend analysis across sessions
    if any(session['data_found'] for session in results['session_comparisons'].values()):
        trends = analyze_trends(results['session_comparisons'])
        results['trend_analysis'] = trends
    
    return results

def analyze_trends(session_comparisons: Dict) -> Dict:
    """
    Analyze trends across sessions for various metrics.
    """
    trends = {
        'activity_trends': {},
        'metric_trends': {},
        'overall_observations': []
    }
    
    # Get sessions with data
    valid_sessions = {
        date: data for date, data in session_comparisons.items() 
        if data['data_found']
    }
    
    if len(valid_sessions) < 2:
        trends['overall_observations'].append(
            "Insufficient sessions for trend analysis"
        )
        return trends
    
    # Analyze numeric metrics trends
    for session_data in valid_sessions.values():
        for metric, values in session_data['numeric_metrics'].items():
            if metric not in trends['metric_trends']:
                trends['metric_trends'][metric] = []
            trends['metric_trends'][metric].append(values['mean'])
    
    # Calculate trend directions
    for metric, values in trends['metric_trends'].items():
        if len(values) >= 2:
            first_val = values[0]
            last_val = values[-1]
            change_pct = ((last_val - first_val) / first_val) * 100
            
            trends['overall_observations'].append(
                f"{metric}: {'increased' if change_pct > 0 else 'decreased'} "
                f"by {abs(change_pct):.1f}% across sessions"
            )
    
    return trends

class PlayerSessionComparisonInput(BaseModel):
    """Input schema for comparing a player's performance across sessions."""
    player_name: str = Field(..., description="Name of the player to analyse")

class ComparePlayerSessions(BaseTool):
    """Tool for comparing a player's performance across multiple sessions."""
    name: str = Field("compare_player_sessions", description="Name of the tool")
    description: str = Field(
        """Compare a player's performance metrics across multiple training sessions, including:
        - Activity type distribution changes
        - Trends in duration, distance, magnitude
        - Changes in metabolic power and dynamic stress load
        - Time-based metric variations
        - Overall performance trends
        
        Input should be the name of the player to analyse across sessions.
        """,
        description="Description of what the tool does"
    )
    args_schema: Type[BaseModel] = Field(PlayerSessionComparisonInput, description="Schema for tool arguments")
    session_data: Dict[str, pd.DataFrame] = Field(default_factory=dict, exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _run(self, player_name: str) -> str:
        """Execute the comparison of a player across sessions."""
        try:
            if not self.session_data:
                return "No session data available for comparison"
            
            comparison_results = compare_player_sessions(
                self.session_data, 
                player_name
            )
            
            # Format the results into a readable string
            output_parts = [
                f"Analysis of {player_name}'s performance across "
                f"{comparison_results['sessions_analysed']} sessions:"
            ]
            
            # Process each session
            for session_date, stats in comparison_results['session_comparisons'].items():
                output_parts.append(f"\n\nSession: {session_date}")
                
                if not stats['data_found']:
                    output_parts.append(stats['message'])
                    continue
                
                output_parts.append(f"Total Activities: {stats['total_activities']}")
                
                # Activity distribution
                output_parts.append("\nActivity Distribution:")
                for activity, details in stats['activity_distribution'].items():
                    output_parts.append(
                        f"- {activity}: {details['count']} "
                        f"({details['percentage']:.1f}%)"
                    )
                
                # Numeric metrics
                output_parts.append("\nPerformance Metrics:")
                for metric, values in stats['numeric_metrics'].items():
                    output_parts.append(
                        f"- {metric}:"
                        f"\n  Average: {values['mean']:.2f}"
                        f"\n  Min: {values['min']:.2f}"
                        f"\n  Max: {values['max']:.2f}"
                    )
                
                # Time metrics
                output_parts.append("\nTime Metrics:")
                for metric, value in stats['time_metrics'].items():
                    output_parts.append(f"- {metric}: {value}")
            
            # Add trend analysis
            if 'trend_analysis' in comparison_results:
                output_parts.append("\n\nTrend Analysis:")
                for observation in comparison_results['trend_analysis']['overall_observations']:
                    output_parts.append(f"- {observation}")
            
            return "\n".join(output_parts)
            
        except Exception as e:
            return f"Error comparing player sessions: {str(e)}"

    async def _arun(self, player_name: str) -> str:
        """Async implementation of the tool."""
        return self._run(player_name)
    
class ComparePlayerSessionsDictionary(BaseTool):
    """Tool for comparing a player's performance across multiple sessions, returning data as a dictionary."""
    name: str = Field("compare_player_sessions_dict", 
                     description="Name of the tool")
    description: str = Field(
        """Compare a player's performance metrics across multiple training sessions, returning results as a dictionary.
        Returns the most recent session's data in a structured format including:
        - Activity type distribution
        - Duration metrics
        - Distance metrics
        - Magnitude metrics
        - Metabolic power metrics
        - Dynamic stress load metrics
        Input should be the name of the player to analyse.
        """,
        description="Description of what the tool does"
    )
    args_schema: Type[BaseModel] = Field(PlayerSessionComparisonInput, 
                                       description="Schema for tool arguments")
    return_direct: bool = True
    session_data: Dict[str, pd.DataFrame] = Field(default_factory=dict, exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _run(self, player_name: str) -> dict:
        """Execute the comparison and return dictionary format."""
        try:
            if not self.session_data:
                return {}
            
            comparison_results = compare_player_sessions(
                self.session_data, 
                player_name
            )
            
            # Get the most recent session's data
            latest_session = list(comparison_results['session_comparisons'].items())[-1][1]
            
            if not latest_session['data_found']:
                return {}
            
            # Format into the structure expected by our RAG system
            metrics = {
                "player_metrics": latest_session['numeric_metrics'],
                "Activity_Distribution": {
                    activity: details['percentage'] 
                    for activity, details in latest_session['activity_distribution'].items()
                },
                "Time_Metrics": latest_session['time_metrics']
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error comparing player sessions: {str(e)}")
            return {}

    async def _arun(self, player_name: str) -> dict:
        """Async implementation of the tool."""
        return self._run(player_name)