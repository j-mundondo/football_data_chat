import pandas as pd
from langchain.tools import tool
from typing import List, Dict, Optional, Type, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

class PlayerSessionComparisonInput(BaseModel):
    """Input schema for comparing a player's performance across sessions."""
    player_name: str = Field(..., description="Name of the player to analyse")



def analyse_trends(session_comparisons: Dict) -> Dict:
    """
    analyse trends across sessions for various metrics.
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

    
    # analyse numeric metrics trends
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
        trends = analyse_trends(results['session_comparisons'])
        results['trend_analysis'] = trends
    
    return results

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
            
            # Return full results including all sessions and trends
            metrics = {
                "player_name": player_name,
                "total_sessions": len(comparison_results['session_comparisons']),
                "sessions": {
                    date: {
                        "player_metrics": session['numeric_metrics'],
                        "Activity_Distribution": {
                            activity: details['percentage'] 
                            for activity, details in session['activity_distribution'].items()
                        },
                        "Time_Metrics": session['time_metrics']
                    }
                    for date, session in comparison_results['session_comparisons'].items()
                    if session['data_found']
                }
            }
            
            # Add trend analysis if available
            if 'trend_analysis' in comparison_results:
                metrics['trends'] = comparison_results['trend_analysis']
            
            return metrics
                
        except Exception as e:
            print(f"Error comparing player sessions: {str(e)}")
            return {}    
    # def _run(self, player_name: str) -> dict:
    #     """Execute the comparison and return dictionary format."""
    #     try:
    #         if not self.session_data:
    #             return {}
            
    #         comparison_results = compare_player_sessions(
    #             self.session_data, 
    #             player_name
    #         )
            
    #         # Get the most recent session's data
    #         latest_session = list(comparison_results['session_comparisons'].items())[-1][1]
            
    #         if not latest_session['data_found']:
    #             return {}
            
    #         # Format into the structure expected by our RAG system
    #         metrics = {
    #             "player_metrics": latest_session['numeric_metrics'],
    #             "Activity_Distribution": {
    #                 activity: details['percentage'] 
    #                 for activity, details in latest_session['activity_distribution'].items()
    #             },
    #             "Time_Metrics": latest_session['time_metrics']
    #         }
            
    #         return metrics
            
    #     except Exception as e:
    #         print(f"Error comparing player sessions: {str(e)}")
    #         return {}

    # async def _arun(self, player_name: str) -> dict:
    #     """Async implementation of the tool."""
    #     return self._run(player_name)
    
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
        trends = analyse_trends(results['session_comparisons'])
        results['trend_analysis'] = trends
    
    return results


from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, Type
import pandas as pd

class PlayerSessionMetricsInput(BaseModel):
    """Input schema for getting a player's metrics from a specific session."""
    player_name: str = Field(..., description="Name of the player to analyse")
    session_date: Optional[str] = Field(None, description="Specific date to analyse (format: YYYY-MM-DD). If not provided, will use latest session")

class SingleSessionPlayerMetrics(BaseTool):
    """Tool for getting a player's metrics from a specific session or latest session."""
    name: str = Field("get_player_session_metrics", 
                     description="Name of the tool")
    description: str = Field(
        """Get a player's performance metrics from a specific session date or their latest session if no date provided.
        Returns metrics including:
        - Activity type distribution
        - Duration metrics
        - Distance metrics
        - Magnitude metrics
        - Metabolic power metrics
        - Dynamic stress load metrics
        - Time-based metrics
        Input requires player name and optionally a specific date (YYYY-MM-DD format).
        If no date provided, returns metrics from the most recent session.
        """,
        description="Description of what the tool does"
    )
    args_schema: Type[BaseModel] = Field(PlayerSessionMetricsInput, 
                                       description="Schema for tool arguments")
    return_direct: bool = True
    session_data: Dict[str, pd.DataFrame] = Field(default_factory=dict, exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _run(self, player_name: str, session_date: Optional[str] = None) -> dict:
        """Execute the metrics retrieval and return dictionary format."""
        try:
            if not self.session_data:
                return {"error": "No session data available"}
            
            # Clean the player name
            clean_player_name = player_name.strip().lower()
            
            # Get all available dates
            available_dates = sorted(list(self.session_data.keys()))
            
            if not available_dates:
                return {"error": "No sessions available"}
            
            # Determine which date to use
            if session_date:
                if session_date not in self.session_data:
                    return {
                        "error": f"No data available for date: {session_date}",
                        "available_dates": available_dates
                    }
                target_date = session_date
            else:
                target_date = available_dates[-1]  # Latest date
            
            # Get the session data
            df = self.session_data[target_date]
            df_clean = df.copy()
            df_clean['Player_Clean'] = df_clean['Player'].str.strip().str.lower()
            
            # Filter for the specific player
            player_df = df_clean[df_clean['Player_Clean'] == clean_player_name]
            
            if player_df.empty:
                return {
                    "error": f"No data found for player '{player_name}' on {target_date}",
                    "date": target_date
                }
            
            # Calculate metrics
            numeric_cols = ['Duration', 'Distance', 'Magnitude', 
                          'Avg Metabolic Power', 'Dynamic Stress Load']
            
            # Activity distribution
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
            
            # Time metrics calculation
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
            
            # Return formatted metrics
            return {
                "player_name": player_name,
                "session_date": target_date,
                "metrics": {
                    "player_metrics": numeric_stats,
                    "Activity_Distribution": {
                        activity: float(count/total_activities * 100)
                        for activity, count in activity_dist.items()
                    },
                    "Time_Metrics": time_stats,
                    "total_activities": total_activities
                }
            }
            
        except Exception as e:
            return {"error": f"Error getting player metrics: {str(e)}"}

    async def _arun(self, player_name: str, session_date: Optional[str] = None) -> dict:
        """Async implementation of the tool."""
        return self._run(player_name, session_date)
    

from langchain.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Type
import pandas as pd

class MultiPlayerComparisonInput(BaseModel):
    """Input schema for comparing multiple players' latest session metrics."""
    player_names: List[str] = Field(..., description="List of player names to compare")

class CompareMultiplePlayersLatestSession(BaseTool):
    """Tool for comparing multiple players' performance metrics from their latest sessions."""
    name: str = Field("compare_multiple_players_latest", 
                     description="Name of the tool")
    description: str = Field(
        """Compare performance metrics between multiple players using their most recent sessions.
        Returns comparative metrics including:
        - Activity type distributions
        - Duration metrics
        - Distance metrics
        - Magnitude metrics
        - Metabolic power metrics
        - Dynamic stress load metrics
        - Time-based metrics
        Input should be a list of player names to compare.
        """,
        description="Description of what the tool does"
    )
    args_schema: Type[BaseModel] = Field(MultiPlayerComparisonInput, 
                                       description="Schema for tool arguments")
    return_direct: bool = True
    session_data: Dict[str, pd.DataFrame] = Field(default_factory=dict, exclude=True)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def _run(self, player_names: List[str]) -> dict:
        """Execute the multi-player comparison and return dictionary format."""
        try:
            if not self.session_data:
                return {"error": "No session data available"}
            
            # Get latest session date
            latest_date = max(self.session_data.keys())
            latest_df = self.session_data[latest_date]
            
            # Clean player names
            clean_player_names = [name.strip().lower() for name in player_names]
            
            # Results dictionary
            results = {
                "session_date": latest_date,
                "players_compared": len(player_names),
                "player_comparisons": {},
                "comparative_analysis": {}
            }
            
            # Process each player
            for player_name, clean_name in zip(player_names, clean_player_names):
                # Clean the DataFrame player names
                latest_df['Player_Clean'] = latest_df['Player'].str.strip().str.lower()
                
                # Filter for the specific player
                player_df = latest_df[latest_df['Player_Clean'] == clean_name]
                
                if player_df.empty:
                    results["player_comparisons"][player_name] = {
                        "data_found": False,
                        "message": f"No data found for player: {player_name}"
                    }
                    continue
                
                # Calculate metrics
                numeric_cols = ['Duration', 'Distance', 'Magnitude', 
                              'Avg Metabolic Power', 'Dynamic Stress Load']
                
                # Activity distribution
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
                
                # Time metrics calculation
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
                
                # Store player results
                results["player_comparisons"][player_name] = {
                    "data_found": True,
                    "total_activities": total_activities,
                    "activity_distribution": {
                        activity: float(count/total_activities * 100)
                        for activity, count in activity_dist.items()
                    },
                    "numeric_metrics": numeric_stats,
                    "time_metrics": time_stats
                }
            
            # Generate comparative analysis
            valid_players = {
                name: data for name, data in results["player_comparisons"].items()
                if data["data_found"]
            }
            
            if len(valid_players) >= 2:
                # Compare numeric metrics
                for metric in numeric_cols:
                    metric_values = {
                        name: data["numeric_metrics"][metric]["mean"]
                        for name, data in valid_players.items()
                        if metric in data["numeric_metrics"]
                    }
                    
                    if metric_values:
                        max_player = max(metric_values.items(), key=lambda x: x[1])
                        min_player = min(metric_values.items(), key=lambda x: x[1])
                        
                        results["comparative_analysis"][metric] = {
                            "highest": {
                                "player": max_player[0],
                                "value": max_player[1]
                            },
                            
                            "lowest": {
                                "player": min_player[0],
                                "value": min_player[1]
                            },
                            "difference_percentage": ((max_player[1] - min_player[1]) / min_player[1]) * 100
                        }
            
            return results
            
        except Exception as e:
            return {"error": f"Error comparing players: {str(e)}"}

    async def _arun(self, player_names: List[str]) -> dict:
        """Async implementation of the tool."""
        return self._run(player_names)