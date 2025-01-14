from datetime import datetime, timedelta
import pandas as pd

from datetime import datetime, timedelta

def standardise_start_times(df):
    """
    Standardizes start times using end time and duration information.
    
    Args:
    df: DataFrame with 'Start Time', 'End Time', and 'Duration' columns
    
    Returns:
    DataFrame with standardized 'Start Time' column
    """
    def parse_duration(duration_str):
        # Convert duration string (MM:SS.s) to seconds
        minutes, seconds = duration_str.split(':')
        return float(minutes) * 60 + float(seconds)
    
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Convert end times to datetime
    df['End Time'] = pd.to_datetime(df['Start Time'].str[:11] + df['End Time'], 
                                  format='%d/%m/%Y %H:%M:%S')
    
    # Convert durations to timedelta
    durations = pd.to_timedelta(df['Duration'].apply(parse_duration), unit='s')
    
    # Calculate precise start times by subtracting duration from end time
    #df['Derived Start Time'] = df['End Time'] - durations
    df.insert(3, 'Calculated Start Time', df['End Time'] - durations)
    if 'Unnamed: 0' in df.columns:
        df=df.drop('Unnamed: 0', axis=1)
    df = df.reset_index(drop=True)
    return df

def add_time_since_last_any_action(df):
    """
    Adds a column calculating time between any consecutive high intensity actions
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Calculate time difference between end of previous action and start of current action
    df['Placeholder'] = (df['Calculated Start Time'].shift(-1) - df['End Time']).shift()

    df.insert(6, 'Time Since Last Any Action', df['Placeholder'].apply(
        lambda x: f"{int(x.total_seconds()//60):02d}:{x.total_seconds()%60:04.1f}" if pd.notna(x) else None
    ))
   # display(df.head())
    df=df.drop('Placeholder', axis=1)
    return df

def full_preprocess(df):
    df = standardise_start_times(df)
    df = add_time_since_last_any_action(df)
    return df