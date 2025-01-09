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

    def get_all_tools():
        return  [
    most_common_event_sequences,
    consecutive_action_frequency,
    analyze_actions_after_distance,
    action_frequency_with_distance]
    return get_all_tools()