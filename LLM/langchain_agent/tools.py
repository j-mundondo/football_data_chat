from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain.agents import AgentType
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.youtube.search import YouTubeSearchTool
import openai
import pandas as pd
from langchain.agents import Tool
from pathlib import Path
from langchain_core.tools import tool
from typing import Dict, List
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains.sequential import SimpleSequentialChain

#@tool
from langchain.tools import tool
import pandas as pd
def create_tools(df):
    @tool
    def most_common_event_sequences(
        n_actions: int = 3,
        max_time_between: float = 10.0
    ) -> Dict:
        """
        Find most common sequences of high-intensity activities in the DataFrame.
        
        Args:
            n_actions: Number of consecutive actions to look for in sequences (default: 3)
            max_time_between: Maximum time in seconds between actions to be considered a sequence (default: 10.0)
        
        Returns:
            Dictionary containing sequences, their counts, and percentages.

        Example Questions Where this function would be useful : 
            * What are the most common 2 action sequences of high-intensity activities in 5 seconds timespans?      
        """
        def parse_time_to_seconds(time_str):
            if pd.isna(time_str):
                return 0
            minutes, seconds = time_str.split(':')
            return float(minutes) * 60 + float(seconds)
        
        # Note: Using df from outer scope
        sequences = []
        current_sequence = []
        
        time_seconds = df['Time Since Last Any Action'].apply(parse_time_to_seconds)
        
        for i in range(len(df)):
            time_since_last = time_seconds.iloc[i]
            
            if time_since_last > max_time_between:
                current_sequence = []
            
            current_sequence.append(df['High Intensity Activity Type'].iloc[i])
            
            if len(current_sequence) >= n_actions:
                sequences.append(tuple(current_sequence[-n_actions:]))
        
        sequence_counts = pd.Series(sequences).value_counts()
        
        return {
            'sequences': list(sequence_counts.index),
            'counts': list(sequence_counts.values),
            'percentages': list((sequence_counts.values / len(sequences) * 100).round(2))
        }
    #most_common_event_sequences(df, n_actions=4)

    @tool
    def consecutive_action_frequency(
    actions: List[str],
    max_time_between: float = 5.0
    ) -> Dict:
        """
        Calculate how often a specific sequence of high-intensity activities occurs in exact order within a given time window.

        Args:
            actions: List of action types in the specific order to look for (e.g. ['Sprint', 'Deceleration'])
            max_time_between: Maximum seconds between actions to be considered part of the same sequence (default: 5.0)

        Returns:
            Dictionary containing:
            - sequence: String showing the sequence analyzed (e.g. "Sprint → Deceleration")
            - total_occurrences: Number of times this exact sequence occurred
            - total_opportunities: Number of times the first action occurred
            - percentage_of_opportunities: Percentage of times first action led to full sequence
            - average_time_between: Average time between actions in successful sequences

        Example Questions Where this function would be useful:
            * How often do players decelerate immediately after sprinting?
            * What percentage of accelerations are followed by sprints within 2 seconds?
            * When a player does a sprint, how likely are they to decelerate within 3 seconds?
            * What's the typical time gap between an acceleration and a sprint when they occur together?
            * Out of all sprints, what percentage lead to a deceleration within 1 second?
        """
        def parse_time_to_seconds(time_str):
            if pd.isna(time_str):
                return 0
            minutes, seconds = time_str.split(':')
            return float(minutes) * 60 + float(seconds)
            
        time_seconds = df['Time Since Last Any Action'].apply(parse_time_to_seconds)
            
        sequence_count = 0
        total_starts = 0
        sequences_found = []
            
        # Count how many times first action appears (total opportunities)
        total_starts = df[df['High Intensity Activity Type'] == actions[0]].shape[0]
            
        # Look for complete sequences
        for i in range(len(df) - len(actions) + 1):
            if df['High Intensity Activity Type'].iloc[i] == actions[0]:
                valid_sequence = True
                sequence_times = []
                    
                # Check if subsequent actions match in order and time constraints
                for j in range(1, len(actions)):
                    if (i + j >= len(df) or 
                        df['High Intensity Activity Type'].iloc[i + j] != actions[j] or 
                        time_seconds.iloc[i + j] > max_time_between):
                        valid_sequence = False
                        break
                    sequence_times.append(time_seconds.iloc[i + j])
                    
                if valid_sequence:
                    sequence_count += 1
                    sequences_found.append({
                        'start_index': i,
                        'times_between': sequence_times
                    })
            
        return {
            'sequence': ' → '.join(actions),
            'total_occurrences': sequence_count,
            'total_opportunities': total_starts,
            'percentage_of_opportunities': float((sequence_count / total_starts * 100 if total_starts > 0 else 0)),
            'average_time_between': sum([sum(s['times_between'])/len(s['times_between']) 
                                        for s in sequences_found])/len(sequences_found) if sequences_found else 0
        }

    @tool
    def analyze_actions_after_distance(
    initial_action: str,
    distance_threshold: float,
    max_time_between: float = 2.0
    ) -> Dict:
        """
        Analyze what actions commonly follow specific high-intensity movements that exceed a distance threshold.

        Args:
            initial_action: The type of action to filter for (e.g., 'Sprint', 'Acceleration')
            distance_threshold: Minimum distance in meters for the initial action to be considered
            max_time_between: Maximum seconds to consider an action as "following" (default: 2.0)

        Returns:
            Dictionary containing:
            - following_actions: List of actions that occur after the filtered initial action
            - counts: Number of times each following action occurs
            - percentages: Percentage distribution of following actions

        Example Questions Where this function would be useful:
            * For sprints greater than 40m, what actions typically follow?
            * After long accelerations (>5m), how often do players transition to sprints?
            * What's the most common follow-up action after extended sprints?
            * For accelerations over 10m, do players tend to decelerate or maintain speed?
            * What percentage of long sprints (>30m) are followed by decelerations?
            * After high-distance movements, what patterns emerge in subsequent actions?
        """
        # First filter for the initial action with distance threshold
        filtered_df = df[
            (df['High Intensity Activity Type'] == initial_action) & 
            (df['Distance'] > distance_threshold)
        ]
        
        following_actions = []
        
        # For each qualifying initial action, find the next action
        for idx in filtered_df.index:
            if idx + 1 < len(df):  # Ensure there is a next action
                time_str = df['Time Since Last Any Action'].iloc[idx + 1]
                if pd.notna(time_str):
                    # Parse time to seconds
                    minutes, seconds = time_str.split(':')
                    time_between = float(minutes) * 60 + float(seconds)
                    
                    if time_between <= max_time_between:
                        following_actions.append(df['High Intensity Activity Type'].iloc[idx + 1])
        
        # Calculate frequencies
        if following_actions:
            frequencies = pd.Series(following_actions).value_counts()
            percentages = (frequencies / len(following_actions) * 100).round(2)
            
            return {
                'following_actions': list(frequencies.index),
                'counts': list(frequencies.values),
                'percentages': list(percentages.values)
            }
        else:
            return {
                'following_actions': [],
                'counts': [],
                'percentages': []
            }
    
    @tool
    def action_frequency_with_distance(
        action_type: str,
        distance_threshold: float,
        greater_than_or_less_than: str = 'greater'
    ) -> Dict:
            """
            Analyze how often specific high-intensity actions occur that meet certain distance criteria.

            Args:
                action_type: Type of movement to analyze (e.g., 'Sprint', 'Acceleration', 'Deceleration')
                distance_threshold: Distance threshold in meters to filter actions
                greater_than_or_less_than: Whether to look for actions 'greater' or 'less' than threshold (default: 'greater')

            Returns:
                Dictionary containing:
                - action_type: The type of action analyzed
                - distance_criteria: Description of the distance filter applied
                - total_occurrences: Number of actions meeting the criteria
                - percentage_of_all_actions: What percentage these actions make up of all activities
                - percentage_of_action_type: What percentage these actions make up of this specific action type
                - average_time_between: Mean time between these actions. This is in the formate MM:SS:MSMS
                - average_distance: Average distance covered in these actions

            Example Questions Where this function would be useful:
                * How often do players complete sprints longer than 30 meters?
                * What percentage of accelerations are shorter than 5 meters?
                * How frequent are long-distance sprints in the data?
                * What's the typical time gap between sprints over 20 meters?
                * How many times do players perform short bursts (<10m) versus extended movements?
                * What portion of all movements are high-distance accelerations?
                * How often do players need to sprint distances greater than 40 meters?
            """
            # Total number of this action type
            total_actions = df[df['High Intensity Activity Type'] == action_type].shape[0]
            
            # Filter by distance
            if greater_than_or_less_than == 'greater':
                filtered_actions = df[
                    (df['High Intensity Activity Type'] == action_type) & 
                    (df['Distance'] > distance_threshold)
                ]
            else:
                filtered_actions = df[
                    (df['High Intensity Activity Type'] == action_type) & 
                    (df['Distance'] < distance_threshold)
                ]
            
            filtered_count = len(filtered_actions)
            
            # Calculate average time between these actions
            if filtered_count > 1:
                def parse_time_to_seconds(time_str):
                    if pd.isna(time_str):
                        return 0
                    minutes, seconds = time_str.split(':')
                    return float(minutes) * 60 + float(seconds)
                    
                times_between = filtered_actions['Time Since Last'].apply(parse_time_to_seconds).mean()
                times_between = f"{int(times_between//60):02d}:{(times_between%60):05.2f}"
            else:
                times_between = None
            
            return {
                'action_type': action_type,
                'distance_criteria': f"{greater_than_or_less_than} than {distance_threshold}m",
                'total_occurrences': filtered_count,
                'percentage_of_all_actions': float((filtered_count / len(df) * 100)),
                'percentage_of_action_type': float((filtered_count / total_actions * 100)) if total_actions > 0 else 0,
                'average_time_between': times_between,
                'average_distance': float(filtered_actions['Distance'].mean()) if filtered_count > 0 else 0
            }

    @tool
    def multiple_actions_in_period(
    action_type: str,
    num_actions: int = 2,
    period_type: str = 'short',
    custom_time: float = None,
    use_long_actions: bool = False
    ) -> Dict:
        """
        Analyze occurrences of multiple similar actions within specified time periods.
        
        Args:
            action_type: Type of action to analyze (e.g., 'Sprint', 'Acceleration')
            num_actions: Number of actions to look for within the period (default: 2)
            period_type: 'short' (<15s) or 'long' (>15s) (default: 'short')
            custom_time: Optional custom time period in seconds (overrides period_type)
            use_long_actions: Whether to only consider actions flagged as 'long' in the data

        Returns:
            Dictionary containing:
            - action_details: Description of what was analyzed
            - total_occurrences: Number of times this pattern occurred
            - average_time_between: Average time between consecutive actions
            - time_distribution: Distribution of time gaps between actions

        Example Questions Where this function would be useful:
            * How often do players have to complete multiple sprints in a short period?
            * Are there instances of 3 accelerations within 10 seconds?
            * How often do long sprints occur close together?
            * What's the typical gap between repeated high-intensity actions?
        """
        def parse_time_to_seconds(time_str):
            if pd.isna(time_str):
                return 0
            minutes, seconds = time_str.split(':')
            return float(minutes) * 60 + float(seconds)

        # Filter for specified action type and long flag if requested
        action_df = df[df['High Intensity Activity Type'] == action_type].copy()
        if use_long_actions:
            action_df = action_df[action_df['Long_sprint'] == 1]
        
        sequences = []
        current_sequence = []
        times_between = []
        
        for i in range(len(action_df)):
            time_since_last = parse_time_to_seconds(action_df.iloc[i]['Time Since Last'])
            
            # Determine if this action should be included based on time criteria
            if custom_time is not None:
                include_action = time_since_last < custom_time if period_type == 'short' else time_since_last > custom_time
            else:
                include_action = time_since_last < 15 if period_type == 'short' else time_since_last > 15
            
            if not include_action:
                if len(current_sequence) >= num_actions:
                    sequences.append(tuple(current_sequence))
                current_sequence = [i]
            else:
                if current_sequence:  # If we're building a sequence
                    times_between.append(time_since_last)
                current_sequence.append(i)
            
            # Check if we've completed a sequence
            if len(current_sequence) >= num_actions:
                sequences.append(tuple(current_sequence))
                
        # Calculate statistics
        total_sequences = len(sequences)
        
        if total_sequences > 0:
            avg_time = sum(times_between) / len(times_between) if times_between else 0
            
            # Calculate time distribution
            time_dist = pd.Series(times_between).describe()
            
            threshold_desc = f"< {custom_time if custom_time is not None else 15}s" if period_type == 'short' else f"> {custom_time if custom_time is not None else 15}s"
            
            return {
                'action_details': (f"{num_actions}x {action_type}" + 
                                    (" (long only)" if use_long_actions else "") + 
                                    f" with gaps {threshold_desc}"),
                'total_occurrences': total_sequences,
                'average_time_between': f"{int(avg_time//60):02d}:{(avg_time%60):05.2f}",
                'time_distribution': {
                    'min': float(time_dist['min']),
                    'max': float(time_dist['max']),
                    '25%': float(time_dist['25%']),
                    '75%': float(time_dist['75%']),
                    'median': float(time_dist['50%'])
                }
            }
        else:
            threshold_desc = f"< {custom_time if custom_time is not None else 15}s" if period_type == 'short' else f"> {custom_time if custom_time is not None else 15}s"
            return {
                'action_details': f"No sequences of {num_actions}x {action_type} found with gaps {threshold_desc}",
                'total_occurrences': 0,
                'average_time_between': None,
                'time_distribution': {}
            }

    @tool
    def sequence_ending_with_action(
        final_action: str,
        sequence_length: int = 3,
        max_time_between: float = 10.0,
        final_must_be_long: bool = False
    ) -> Dict:
            """
            Analyze sequences of high-intensity activities that end with a specific action.
            
            Args:
                final_action: The action that should end the sequence (e.g., 'Sprint')
                sequence_length: Total number of actions in the sequence, including final action (default: 3)
                max_time_between: Maximum time (in seconds) allowed between actions in sequence
                final_must_be_long: Whether the final action must be flagged as 'long' (default: False)

            Returns:
                Dictionary containing:
                - sequence_details: Description of sequence criteria
                - common_patterns: Most common sequences found, ending with specified action
                - total_occurrences: Number of qualifying sequences found
                - average_time_total: Average time to complete full sequence
                - sequence_distribution: Frequency of different patterns found

            Example Questions Where this function would be useful:
                * How often do players have to go through a series of high intensity activities and then complete a long sprint?
                * What typically precedes a long sprint?
                * What patterns of activity commonly lead to decelerations?
                * How many times do players do multiple high-intensity actions before a sprint?
            """
            def parse_time_to_seconds(time_str):
                if pd.isna(time_str):
                    return 0
                minutes, seconds = time_str.split(':')
                return float(minutes) * 60 + float(seconds)
            
            sequences = []
            sequence_times = []
            current_sequence = []
            current_times = []
            
            # Convert times to seconds
            time_seconds = df['Time Since Last Any Action'].apply(parse_time_to_seconds)
            
            for i in range(len(df)):
                current_action = df['High Intensity Activity Type'].iloc[i]
                time_since_last = time_seconds.iloc[i]
                
                # Check if this is the final action we're looking for
                is_final_action = (current_action == final_action and 
                                (not final_must_be_long or df['Long_sprint'].iloc[i] == 1))
                
                if time_since_last > max_time_between:
                    current_sequence = []
                    current_times = []
                
                current_sequence.append(current_action)
                if time_since_last > 0:  # Don't add time for first action
                    current_times.append(time_since_last)
                
                # If we have the right length and ends with final action
                if (len(current_sequence) >= sequence_length and 
                    is_final_action and 
                    current_sequence[-1] == final_action):
                    sequences.append(tuple(current_sequence[-sequence_length:]))
                    sequence_times.append(sum(current_times[-sequence_length+1:]))  # +1 because first action has no time
            
            total_sequences = len(sequences)
            
            if total_sequences > 0:
                # Get sequence frequencies
                sequence_counts = pd.Series(sequences).value_counts()
                
                return {
                    'sequence_details': (f"Sequences of {sequence_length} actions ending in {final_action}" +
                                    (" (long)" if final_must_be_long else "")),
                    'common_patterns': [
                        {
                            'sequence': ' → '.join(seq),
                            'count': count,
                            'percentage': float((count/total_sequences * 100))
                        }
                        for seq, count in sequence_counts.head(5).items()
                    ],
                    'total_occurrences': total_sequences,
                    'average_time_total': f"{(sum(sequence_times)/len(sequence_times)):.2f}s",
                    'sequence_distribution': {
                        'unique_patterns': len(sequence_counts),
                        'most_common_count': int(sequence_counts.iloc[0]),
                        'least_common_count': int(sequence_counts.iloc[-1])
                    }
                }
            else:
                return {
                    'sequence_details': (f"No sequences of {sequence_length} actions ending in {final_action}" +
                                    (" (long)" if final_must_be_long else "")),
                    'common_patterns': [],
                    'total_occurrences': 0,
                    'average_time_total': None,
                    'sequence_distribution': {}
                }
    @tool
    def count_specific_actions(relevant_colum: str,action_type: str = None) -> Dict:
        """
        Count occurrences of a specific action type or all actions.
        
        Args:
            action_type: Specific action to count (e.g., 'Sprint', 'Acceleration'). If None, counts all action types.
            relevant_colum : The column the question is being asked about
            
        Returns:
            Dictionary with counts and percentages
            
        Example Questions:
            * How many sprints are there?
            * What is the total number of accelerations?
            * How many times did players sprint?
            * Count all decelerations
        """
        if action_type:
            count = len(df[df[relevant_colum] == action_type])
            total = len(df)
            return {
                'action_type': action_type,
                'count': int(count),
                'percentage': float((count/total * 100).round(2)),
                'total_actions': total
            }
        else:
            counts = df[relevant_colum].value_counts()
            return {
                'counts_by_type': {k: int(v) for k, v in counts.items()},
                'total_actions': int(len(df))
            }

    @tool
    def get_numeric_column_stats(relevant_colum: str,column_name: str, action_type: str = None) -> Dict:
        """
        Get statistics for a numeric column, optionally filtered by action type.
        
        Args:
            column_name: Column to analyze (e.g., 'Distance', 'Duration_seconds')
            action_type: Optional filter for specific action type
            relevant_colum : The column the question is being asked about
            
        Returns:
            Dictionary with statistical summary
            
        Example Questions:
            * What's the average sprint distance?
            * What's the maximum acceleration duration?
            * How long do sprints typically last?
            * What's the highest magnitude for decelerations?
        """
        if action_type:
            filtered_df = df[df[relevant_colum] == action_type]
            if len(filtered_df) == 0:
                return {'error': f'No data found for action type: {action_type}'}
            stats = filtered_df[column_name].describe()
        else:
            stats = df[column_name].describe()
        
        return {
            'column': column_name,
            'action_type': action_type if action_type else 'all',
            'min': float(stats['min']),
            'max': float(stats['max']),
            'mean': float(stats['mean']),
            'median': float(stats['50%']),
            'std': float(stats['std']),
            'count': int(stats['count'])
        }

    @tool
    def find_most_common_actions(relevant_colum: str,top_n: int = 3) -> Dict:
        """
        Find the most frequent action types.
        
        Args:
            top_n: Number of top actions to return (default: 3)
            relevant_colum : The column the question is being asked about
            
        Returns:
            Dictionary with frequency analysis
            
        Example Questions:
            * What's the most common action?
            * What are the top 3 most frequent activities?
            * Which high intensity actions happen most often?
            * Show me the action frequency distribution
        """
        counts = df[relevant_colum].value_counts()
        top_actions = counts.head(top_n)
        
        return {
            'top_actions': {
                action: {
                    'count': int(count),
                    'percentage': float((count/len(df) * 100).round(2))
                }
                for action, count in top_actions.items()
            },
            'total_actions': len(df),
            'unique_actions': len(counts)
        }

    def get_all_tools():
        return [
            most_common_event_sequences,
            consecutive_action_frequency,
            analyze_actions_after_distance,
            action_frequency_with_distance,
            multiple_actions_in_period,
            sequence_ending_with_action,
            count_specific_actions,
            get_numeric_column_stats,
            find_most_common_actions
        ]
    
    return get_all_tools()            
    # @tool
    # def get_total_count(column_name: str = None) -> Dict:
    #     """
    #     Get the total count of rows, optionally filtered by a specific column.
        
    #     Args:
    #         column_name: Optional column name to count non-null values for
            
    #     Returns:
    #         Dictionary with total count and details
            
    #     Example Questions:
    #         * How many total actions are there?
    #         * What is the total number of rows?
    #         * How many non-null values are in [column]?
    #     """
    #     if column_name:
    #         count = df[column_name].count()
    #         return {
    #             'count': int(count),
    #             'column': column_name,
    #             'total_rows': len(df),
    #             'percentage_filled': float((count/len(df) * 100).round(2))
    #         }
    #     return {
    #         'count': len(df),
    #         'details': 'Total number of rows in dataset'
    #     }

    # @tool
    # def get_column_stats(column_name: str) -> Dict:
    #     """
    #     Get basic statistics for a numeric column.
        
    #     Args:
    #         column_name: Name of the column to analyze
            
    #     Returns:
    #         Dictionary with basic statistics
            
    #     Example Questions:
    #         * What is the maximum value in [column]?
    #         * What are the statistics for [column]?
    #         * What's the average [column]?
    #     """
    #     if column_name not in df.columns:
    #         return {'error': f'Column {column_name} not found'}
        
    #     try:
    #         stats = df[column_name].describe()
    #         return {
    #             'column': column_name,
    #             'min': float(stats['min']),
    #             'max': float(stats['max']),
    #             'mean': float(stats['mean']),
    #             'median': float(stats['50%']),
    #             'std': float(stats['std'])
    #         }
    #     except:
    #         return {'error': f'Cannot compute statistics for {column_name}. Ensure it is numeric.'}

    # @tool
    # def get_value_frequencies(
    #     column_name: str,
    #     top_n: int = None,
    #     normalize: bool = False
    # ) -> Dict:
    #     """
    #     Get frequency distribution of values in a column.
        
    #     Args:
    #         column_name: Column to analyze
    #         top_n: Optional limit on number of values to return
    #         normalize: Whether to return percentages instead of counts
            
    #     Returns:
    #         Dictionary with value frequencies
            
    #     Example Questions:
    #         * What is the most common value in [column]?
    #         * How often does each value appear in [column]?
    #         * What's the distribution of values in [column]?
    #     """
    #     if column_name not in df.columns:
    #         return {'error': f'Column {column_name} not found'}
        
    #     value_counts = df[column_name].value_counts(normalize=normalize)
    #     if top_n:
    #         value_counts = value_counts.head(top_n)
            
    #     return {
    #         'column': column_name,
    #         'frequencies': dict(value_counts),
    #         'total_unique_values': len(value_counts),
    #         'most_common': value_counts.index[0],
    #         'least_common': value_counts.index[-1],
    #         'is_percentage': normalize
    #     }
    # def get_all_tools():
    #     return  [
    # most_common_event_sequences,
    # consecutive_action_frequency,
    # analyze_actions_after_distance,
    # action_frequency_with_distance,
    # multiple_actions_in_period,
    # sequence_ending_with_action]
    # return get_all_tools()
    # def get_all_tools():
    #     return [
    #         most_common_event_sequences,
    #         consecutive_action_frequency,
    #         analyze_actions_after_distance,
    #         action_frequency_with_distance,
    #         multiple_actions_in_period,
    #         sequence_ending_with_action,
    #         get_total_count,
    #         get_column_stats,
    #         get_value_frequencies
    #     ]
    
    # return get_all_tools()