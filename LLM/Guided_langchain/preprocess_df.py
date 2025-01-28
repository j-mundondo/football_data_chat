import pandas as pd
import numpy as np

def standardise_start_times(df):
    """
    Standardises start times using end time and duration information.
    Handles NaT values and different date/time formats.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Remove any summary rows (containing 'Average' or similar)
    df = df[~df['Start Time'].astype(str).str.contains('Average', na=False)]
    df = df[~df['End Time'].astype(str).str.contains('Average', na=False)]
    df = df[~df['Start Time'].astype(str).str.contains('Total/Max', na=False)]
    df = df[~df['End Time'].astype(str).str.contains('Total/Max', na=False)]
    if 'Date' in df.columns:
        df = df.dropna(subset=['Date'])
    
    # Drop rows where Start Time or End Time is NaT
    df = df.dropna(subset=['Start Time', 'End Time'])
    
    
    # Clean player names
    df['Player'] = df['Player'].str.strip().str.lower()
    
    def parse_duration(duration_str):
        try:
            # Convert duration string (MM:SS.s) to seconds
            minutes, seconds = duration_str.split(':')
            return float(minutes) * 60 + float(seconds)
        except (ValueError, AttributeError):
            return np.nan
    
    # Different processing based on whether Date column exists
    if 'Date' in df.columns:
        try:
            # Convert Date column to datetime with explicit format
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', dayfirst=True)
            
            # Convert Start Time and End Time to time format
            df['Start Time'] = pd.to_datetime(df['Start Time'], format='%H:%M:%S', errors='coerce')
            df['End Time'] = pd.to_datetime(df['End Time'], format='%H:%M:%S', errors='coerce')
            
            # Combine date and time, handling NaT values
            df['Start Time'] = df.apply(
                lambda row: pd.Timestamp.combine(
                    row['Date'].date(),
                    row['Start Time'].time()
                ) if pd.notna(row['Date']) and pd.notna(row['Start Time']) else pd.NaT,
                axis=1
            )
            
            df['End Time'] = df.apply(
                lambda row: pd.Timestamp.combine(
                    row['Date'].date(),
                    row['End Time'].time()
                ) if pd.notna(row['Date']) and pd.notna(row['End Time']) else pd.NaT,
                axis=1
            )
            
        except Exception as e:
            print(f"Warning: Error processing dates and times: {str(e)}")
            return df
    else:
        try:
            # Use the original logic when Start Time contains the date
            df['End Time'] = pd.to_datetime(
                df['Start Time'].str[:11] + df['End Time'],
                format='%d/%m/%Y %H:%M:%S',
                errors='coerce'
            )
            df['Start Time'] = pd.to_datetime(
                df['Start Time'],
                format='%d/%m/%Y %H:%M',
                errors='coerce'
            )
        except Exception as e:
            print(f"Warning: Failed to parse timestamps: {str(e)}")
            return df
    
    # Convert durations to timedelta, handling invalid values
    df['Duration'] = df['Duration'].apply(parse_duration)
    durations = pd.to_timedelta(df['Duration'], unit='s')
    
    # Calculate precise start times
    df.insert(3, 'Calculated Start Time', df['End Time'] - durations)
    
    # Clean up unnecessary columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    return df.reset_index(drop=True)

def add_time_since_last_any_action(df):
    """
    Adds a column calculating time between any consecutive high intensity actions.
    Handles NaT values and edge cases.
    """
    df = df.copy()
    
    # Calculate time difference between end of previous action and start of current action
    time_diff = (df['Calculated Start Time'].shift(-1) - df['End Time']).shift()
    
    # Format time differences, handling NaN values
    def format_timedelta(td):
        if pd.isna(td):
            return None
        total_seconds = td.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:04.1f}"
    
    df.insert(6, 'Time Since Last Any Action', 
             time_diff.apply(format_timedelta))
    
    return df

def all_player_analysis(df):
    """
    Performs analysis across all players.
    """
    if 'Session' in df.columns and 'Drill' in df.columns:
        df = df.drop(['Session', 'Drill'], axis=1)
    return df

def full_preprocess(df):
    """
    Applies all preprocessing steps in sequence.
    """
    try:
        df = standardise_start_times(df)
        df = add_time_since_last_any_action(df)
        
        if all(col in df.columns for col in ['Session', 'Player']):
            df = all_player_analysis(df)
            
        if 'HIA Type' in df.columns:
            df = df.rename(columns={'HIA Type': 'High Intensity Activity Type'})
        
        print("Preprocessing completed successfully")
        return df
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None