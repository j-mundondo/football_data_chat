from llm import get_llama3_8b_8192, get_70b_8192, get_llama_3dot3_70b_versatile, get_llama_3dot1_8b_instant
from individual_tools import create_tools
from preprocess_df import full_preprocess
import pandas as pd
#from multiplayer_tools import compare_players
#from custom_agents_LG import player_comparison_agent

df = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-15-17 Players-HIAs-Export_.csv")
df = full_preprocess(df)
#print(f"2025-01-15-17 : \n{df['Player'].value_counts()}")

df2 = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-12-16 Players-HIAs-Export.csv")
df2 = full_preprocess(df2)
#print(f"2025-01-16 : \n{df2['Player'].value_counts()}")

df3 = pd.read_csv(r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\2025-01-17-24 Players-HIAs-Export.csv")
df3 = full_preprocess(df3)
#print(f"2025-01-24 : \n{df3['Player'].value_counts()}")

print(df.describe())
print(df2.describe())
print(df3.describe())
print(df.head())
print(df['High Intensity Activity Type'].value_counts())

