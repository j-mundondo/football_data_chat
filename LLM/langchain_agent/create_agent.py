# # from llm import get_llm
# # from tools import create_tools
# # import pandas as pd
# # from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# # def custom_agent(dataframe, memory=None):
# #     """Create a custom agent optimized for Groq"""
# #     llm = get_llm()
# #     tools = create_tools(dataframe)
      
# #     system_prompt = """You are a sports movement analyst. You have access to a pandas DataFrame with athlete performance data.

# #     Simple Rules:
# #     1. For basic counts and frequencies, ALWAYS use the full DataFrame (df) not just the preview
# #     2. Key DataFrame operations:
# #     - Basic counts: Use len(df) for total rows
# #     - Value counts: Use df['column'].value_counts()
# #     - Summary stats: Use df.describe()
# #     3. Use tools only for:
# #     - Time sequences
# #     - Complex patterns
# #     - Advanced statistics

# #     4. When analyzing numeric data:
# #    - First state what operation you'll perform
# #    - Then provide the code in a single clean block
# #    - Format your response like this:
# #      "To find [metric], I will check [column] using:
# #      df['column'].max()"

# #     5. For basic operations use:
# #     - Total count: len(df)
# #     - Column max: df['column'].max()
# #     - Column stats: df['column'].describe()
# #     - Frequencies: df['column'].value_counts()

# #     6. Always structure responses as:
# #     STEP 1: State the approach
# #     STEP 2: Show the single code block
# #     STEP 3: Explain what the code will tell us
# #     STEP 4: The Actual answer

# #     Example Response Format:
# #     "To find the highest value in the Dynamic Stress Load:
# #     df['Dynamic Stress Load'].max()
# #     This will give us the maximum stress load across all measurements."

# #     Data Context:
# #     Total rows in dataset: {total_rows}
# #     Preview of data structure:
# #     {df_info}

# #     IMPORTANT: While you only see a preview above, you must ALWAYS operate on the full dataset.
# #     For example:
# #     - ❌ Wrong: Counting rows in preview
# #     - ✅ Right: Using len(df) for total count
# #     - ❌ Wrong: df.head()['column'].value_counts()
# #     - ✅ Right: df['column'].value_counts()

# #     # Add these examples to the prompt
# #     Example Operations:
# #     Q: "How many actions are in the dataset?"
# #     A: I'll use len(df) to get the total count: `print(len(df))`

# #     Q: "What's the most common action?"
# #     A: I'll use value_counts() on the full dataset: `print(df['High Intensity Action'].value_counts())`

# #     Remember: Any operation involving counts, frequencies, or statistics must be performed on the entire DataFrame, not just the preview.
    
# #     You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
# #     """
# #     formatted_prompt = system_prompt.format(
# #         total_rows=len(dataframe),
# #         df_info=str(dataframe.info())
# #     )
# #     # Create the agent with simplified configuration
# #     agent = create_pandas_dataframe_agent(
# #         llm=llm,
# #         df=dataframe,
# #         #number_of_head_rows=dataframe.shape[0],s
# #         extra_tools=tools,
# #         verbose=True,
# #         handle_parsing_errors=True,
# #          allow_dangerous_code=True,
# #         agent_type="tool-calling",
# #         memory=memory,
# #         prefix=formatted_prompt#system_prompt.format(df_info=str(dataframe.info())),
# #     )
    
# #     return agent


# from llm import get_llm
# from tools import create_tools
# import pandas as pd
# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# def custom_agent(dataframe, memory=None):
#     """Create a custom agent optimized for Groq"""
#     llm = get_llm()
#     tools = create_tools(dataframe)
    
#     system_prompt = """You are a sports movement analyst. You have access to a pandas DataFrame with athlete performance data.

#     Simple Rules:
#     1. For basic counts and frequencies, ALWAYS use the full DataFrame (df) not just the preview
#     2. Key DataFrame operations:
#     - Basic counts: Use len(df) for total rows
#     - Value counts: Use df['column'].value_counts()
#     - Summary stats: Use df.describe()
#     3. Use tools only for:
#     - Time sequences
#     - Complex patterns
#     - Advanced statistics

#     4. When analyzing numeric data:
#     - First state what operation you'll perform
#     - Then provide the code in a single clean block
#     - Format your response like this:
#       "To find [metric], I will check [column] using:
#       df['column'].max()"

#     5. For basic operations use:
#     - Total count: len(df)
#     - Column max: df['column'].max()
#     - Column stats: df['column'].describe()
#     - Frequencies: df['column'].value_counts()

#     6. Always structure responses as:
#     STEP 1: State the approach
#     STEP 2: Show the single code block
#     STEP 3: Explain what the code will tell us
#     STEP 4: The Actual answer

#     Example Response Format:
#     "To find the highest value in the Dynamic Stress Load:
#     df['Dynamic Stress Load'].max()
#     This will give us the maximum stress load across all measurements."

#     Data Context:
#     Total rows in dataset: {total_rows}
#     Preview of data structure:
#     {df_info}

#     IMPORTANT: While you only see a preview above, you must ALWAYS operate on the full dataset.
#     For example:
#     - ❌ Wrong: Counting rows in preview
#     - ✅ Right: Using len(df) for total count
#     - ❌ Wrong: df.head()['column'].value_counts()
#     - ✅ Right: df['column'].value_counts()

#     Example Operations:
#     Q: "How many actions are in the dataset?"
#     A: I'll use len(df) to get the total count: `print(len(df))`

#     Q: "What's the most common action?"
#     A: I'll use value_counts() on the full dataset: `print(df['High Intensity Action'].value_counts())`

#     Remember: Any operation involving counts, frequencies, or statistics must be performed on the entire DataFrame, not just the preview.
    
#     You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

#     After executing any query:
#     1. Verify the output exists and makes sense
#     2. If you get an error or no output:
#        - Check column names
#        - Verify the operation
#        - Try again with corrected code
#     3. ALWAYS show the numeric result
#     4. End with "Based on the actual data, [conclusion with numbers]"
    
#     Example of correct response:
#     ```python
#     result = df['column'].value_counts()
#     print(f"Result: [actual results]")
#     ```
#     Result: [actual numbers]
#     Based on the actual data, X has Y occurrences."""

#     formatted_prompt = system_prompt.format(
#         total_rows=len(dataframe),
#         df_info=str(dataframe.info())
#     )

#     # Create the agent with simplified configuration
#     agent = create_pandas_dataframe_agent(
#         llm=llm,
#         df=dataframe,
#         #number_of_head_rows=dataframe.shape[0],
#         extra_tools=tools,
#         verbose=True,
#         handle_parsing_errors=True,
#         allow_dangerous_code=True,
#         agent_type="tool-calling",
#         memory=memory,
#         prefix=formatted_prompt#system_prompt.format(df_info=str(dataframe.info())),
#     )
    
#     return agent

from llm import get_llm
from tools import create_tools
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def custom_agent(dataframe, memory=None):
    """Create a custom agent optimized for Groq"""
    llm = get_llm()
    tools = create_tools(dataframe)
    
    system_prompt = """You are a sports movement analyst. You have access to a pandas DataFrame with athlete performance data.

    Simple Rules:
    1. For basic counts and frequencies, ALWAYS use the full DataFrame (df) not just the preview
    2. Key DataFrame operations:
    - Basic counts: Use len(df) for total rows
    - Value counts: Use df['column'].value_counts()
    - Summary stats: Use df.describe()
    3. Use tools only for:
    - Time sequences
    - Complex patterns
    - Advanced statistics

    4. When analyzing numeric data:
    - First state what operation you'll perform
    - Then provide the code in a single clean block
    - Format your response like this:
      "To find [metric], I will check [column] using:
      df['column'].max()"

    5. For basic operations use:
    - Total count: len(df)
    - Column max: df['column'].max()
    - Column stats: df['column'].describe()
    - Frequencies: df['column'].value_counts()

    6. Always structure responses as:
    STEP 1: State the approach
    STEP 2: Show the single code block
    STEP 3: Explain what the code will tell us
    STEP 4: The Actual answer

    Example Response Format:
    "To find the highest value in the Dynamic Stress Load:
    df['Dynamic Stress Load'].max()
    This will give us the maximum stress load across all measurements."

    Data Context:
    Total rows in dataset: {total_rows}
    Preview of data structure:
    {df_info}

    IMPORTANT: While you only see a preview above, you must ALWAYS operate on the full dataset.
    For example:
    - ❌ Wrong: Counting rows in preview
    - ✅ Right: Using len(df) for total count
    - ❌ Wrong: df.head()['column'].value_counts()
    - ✅ Right: df['column'].value_counts()

    Example Operations:
    Q: "How many actions are in the dataset?"
    A: I'll use len(df) to get the total count: `print(len(df))`

    Q: "What's the most common action?"
    A: I'll use value_counts() on the full dataset: `print(df['High Intensity Action'].value_counts())`

    Remember: Any operation involving counts, frequencies, or statistics must be performed on the entire DataFrame, not just the preview.
    
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again."""

    suffix = """After executing any query:
    1. Verify the output exists and makes sense
    2. If you get an error or no output:
       - Check column names
       - Verify the operation
       - Try again with corrected code
    3. ALWAYS show the numeric result
    4. End with "Based on the actual data, [conclusion with numbers]"
    
    Example of correct response:
    ```python
    result = df['column'].value_counts()
    print(f"Result: [result]")
    ```
    Result: [actual numbers]
    Based on the actual data, X has Y occurrences."""

    formatted_prompt = system_prompt.format(
        total_rows=len(dataframe),
        df_info=str(dataframe.info())
    )

    formatted_suffix = f"{suffix}"

    # Create the agent with simplified configuration
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        #number_of_head_rows=dataframe.shape[0],
        extra_tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        memory=memory,
        prefix=formatted_prompt,
        suffix=formatted_suffix
    )
    
    return agent