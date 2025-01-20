from datetime import datetime, timedelta
import pandas as pd

from datetime import datetime, timedelta

# def standardise_start_times(df):
#     """
#     Standardizes start times using end time and duration information.
    
#     Args:
#     df: DataFrame with 'Start Time', 'End Time', and 'Duration' columns
    
#     Returns:
#     DataFrame with standardized 'Start Time' column
#     """
#     def parse_duration(duration_str):
#         # Convert duration string (MM:SS.s) to seconds
#         minutes, seconds = duration_str.split(':')
#         return float(minutes) * 60 + float(seconds)
    
#     # Create a copy to avoid modifying the original DataFrame
#     df = df.copy()
    
#     # Convert end times to datetime
#     df['End Time'] = pd.to_datetime(df['Start Time'].str[:11] + df['End Time'], 
#                                   format='%d/%m/%Y %H:%M:%S')
    
#     # Convert durations to timedelta
#     durations = pd.to_timedelta(df['Duration'].apply(parse_duration), unit='s')
    
#     # Calculate precise start times by subtracting duration from end time
#     #df['Derived Start Time'] = df['End Time'] - durations
#     df.insert(3, 'Calculated Start Time', df['End Time'] - durations)
#     if 'Unnamed: 0' in df.columns:
#         df=df.drop('Unnamed: 0', axis=1)
#     df = df.reset_index(drop=True)
#     return df
def standardise_start_times(df):
    """
    Standardizes start times using end time and duration information.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Remove any summary rows (containing 'Average' or similar)
    df = df[~df['Start Time'].astype(str).str.contains('Average', na=False)]
    df = df[~df['End Time'].astype(str).str.contains('Average', na=False)]
    df = df[~df['Start Time'].astype(str).str.contains('Total/Max', na=False)]
    df = df[~df['End Time'].astype(str).str.contains('Total/Max', na=False)]
    
    def parse_duration(duration_str):
        # Convert duration string (MM:SS.s) to seconds
        minutes, seconds = duration_str.split(':')
        return float(minutes) * 60 + float(seconds)
    
    # Different processing based on whether Date column exists
    if 'Date' in df.columns:
        # If we have a separate Date column, handle Start Time and End Time as times only
        df['Start Time'] = pd.to_datetime(df['Start Time'], format='%H:%M:%S')
        df['End Time'] = pd.to_datetime(df['End Time'], format='%H:%M:%S')
        
        # Add the date information
        date = pd.to_datetime(df['Date'])
        df['Start Time'] = df.apply(lambda row: pd.Timestamp.combine(
            date[row.name].date(), 
            row['Start Time'].time()), axis=1)
        df['End Time'] = df.apply(lambda row: pd.Timestamp.combine(
            date[row.name].date(), 
            row['End Time'].time()), axis=1)
    else:
        # Use the original logic when Start Time contains the date
        try:
            df['End Time'] = pd.to_datetime(df['Start Time'].str[:11] + df['End Time'], 
                                          format='%d/%m/%Y %H:%M:%S')
            df['Start Time'] = pd.to_datetime(df['Start Time'], 
                                            format='%d/%m/%Y %H:%M')
        except:
            print("Warning: Failed to parse timestamps. Check your date/time formats.")
            return df
    
    # Convert durations to timedelta
    durations = pd.to_timedelta(df['Duration'].apply(parse_duration), unit='s')
    
    # Calculate precise start times
    df.insert(3, 'Calculated Start Time', df['End Time'] - durations)
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
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

def all_player_analysis(df):
    df = df.drop(['Session', 'Drill'], axis=1)
    return df

def full_preprocess(df):
    df = standardise_start_times(df)
    df = add_time_since_last_any_action(df)
    if 'Session' and 'Player' in df.columns:
        df = all_player_analysis(df)
    if 'HIA Type' in df.columns:
        df = df.rename(columns={'HIA Type': 'High Intensity Activity Type'})
    print(df)
    return df