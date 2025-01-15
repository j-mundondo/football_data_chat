from llm import get_llm
from tools import create_tools
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# def custom_agent(dataframe, memory=None):
#     """Create a custom agent optimized for Groq"""
#     llm = get_llm()
#     tools = create_tools(dataframe)
    
#     # Field descriptions
#     field_descriptions = {
#         'High Intensity Activity Type': 'Classification of the movement (Acceleration/Deceleration/Sprint)',
#         'Start Time': 'Timestamp when activity began (HH:MM:SS)',
#         'End Time': 'Timestamp when activity ended (HH:MM:SS)', 
#         'Time Since Last': 'Duration since previous activity of that same type (MM:SS.S)',
#         'Duration': 'Length of activity (MM:SS.S)',
#         'Distance': 'Meters traveled during activity',
#         'Magnitude': 'Peak intensity measure of the activity',
#         'Avg Metabolic Power': 'Average energy expenditure during activity (W/kg)',
#         'Dynamic Stress Load': 'Cumulative stress metric from activity intensity',
#         'Duration_seconds': 'The number of seconds the High Intensity Activity Type in the row lasted as a float',
#         'Long_sprint': 'A binary flag indicating if a movement is a long sprint. 1 indicates it is a long sprint, 0 means it is not.',
#         'Preceding High Intensity Activity Type': 'The type of high intensity activity type that happened before this row.'
#     }
    
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
#       df['column'].max()" then execute the code using python_repl"

#     5. For basic operations use:
#     - Total count: len(df)
#     - Column max: df['column'].max()
#     - Column stats: df['column'].describe()
#     - Frequencies: df['column'].value_counts()

#     6. Always structure responses as:
#     STEP 1: State the approach
#     STEP 2: Show the single code block
#     STEP 3: Explain what the code will tell us
#     STEP 4: Provide the actual, numerical answer

#     Field Descriptions:
#     {field_descriptions}

#     Example Response Format:
#     "To find the highest value in the Dynamic Stress Load:
#     result=df['Dynamic Stress Load'].max()
#     This will give us the maximum stress load across all measurements.
#     result"

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

#     """

#     suffix = """After executing any query, you MUST:
#     1. Show the actual numeric output in your response
#     2. Do NOT use placeholder text like '[actual numbers]' or '[value]'
#     3. Include the exact numbers from your calculation
#     4. End with a clear conclusion using the real numbers
#     5. MAKE SURE YOU DOUBLE CHECK THE OUTPUT SO YOU ENSURE THE RESULTS ARE CORRECT. THERE ARE LIVES AT STAKE.

#     Example format:
#     ```python
#     result = df['High Intensity Activity Type'].value_counts()
#     print(f"Result: result")
#     print(result)
#     ```
    
#     The output numbers should be shown exactly as they appear"""
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
#       df['column'].max()" then execute the code using python_repl"

#     5. For basic operations use:
#     - Total count: len(df)
#     - Column max: df['column'].max()
#     - Column stats: df['column'].describe()
#     - Frequencies: df['column'].value_counts()

#     6. Always structure responses as:
#     STEP 1: State the approach
#     STEP 2: Show the single code block
#     STEP 3: Explain what the code will tell us
#     STEP 4: Provide the actual, numerical answer

#     Field Descriptions:
#     {field_descriptions}

#     Example Response Format:
#     "To find the highest value in the Dynamic Stress Load:
#     result=df['Dynamic Stress Load'].max()
#     This will give us the maximum stress load across all measurements.
#     result"

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

#     """

#     suffix = """After executing any query, you MUST:
#     1. Show the actual numeric output in your response
#     2. Do NOT use placeholder text like '[actual numbers]' or '[value]'
#     3. Include the exact numbers from your calculation
#     4. End with a clear conclusion using the real numbers
#     5. MAKE SURE YOU DOUBLE CHECK THE OUTPUT SO YOU ENSURE THE RESULTS ARE CORRECT. THERE ARE LIVES AT STAKE.

#     Example format:
#     ```python
#     result = df['High Intensity Activity Type'].value_counts()
#     print(f"Result: result")
#     print(result)
#     ```
    
#     The output numbers should be shown exactly as they appear"""
#     formatted_prompt = system_prompt.format(
#         total_rows=len(dataframe),
#         df_info=str(dataframe.info()),
#         field_descriptions="\n".join([f"- {k}: {v}" for k, v in field_descriptions.items()])
#     )

#     formatted_suffix = f"{suffix}"

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
#         prefix=formatted_prompt,suffix=formatted_suffix
#     )
    
#     return agent
def custom_agent(dataframe, memory=None):
    """Create a custom agent optimized for Groq"""
    llm = get_llm()
    tools = create_tools(dataframe)
        # Field descriptions
    field_descriptions = {
        'High Intensity Activity Type': 'Classification of the movement (Acceleration/Deceleration/Sprint). This is the field used when referring to an activity, action, HIA, high intensity activity etc.',
        'Start Time': 'Timestamp when activity or action began (HH:MM:SS)',
        'End Time': 'Timestamp when activityor action ended (HH:MM:SS)', 
        'Time Since Last': 'Duration since previous activity or action of that same type (MM:SS.S)',
        'Duration': 'Length of activity (MM:SS.S) i.e. How long that activity lasted',
        'Distance': 'Meters traveled during activity',
        'Magnitude': 'Peak intensity measure of the activity',
        'Avg Metabolic Power': 'Average energy expenditure during activity (W/kg)',
        'Dynamic Stress Load': 'Cumulative stress metric from activity intensity',
        'Duration_seconds': 'The number of seconds the High Intensity Activity Type in the row lasted as a float',
        'Long_sprint': 'A binary flag indicating if a movement is a long sprint. 1 indicates it is a long sprint, 0 means it is not.',
        'Preceding High Intensity Activity Type': 'The type of high intensity activity type that happened before the current entry.'
    }
    system_prompt = """You are a sports data analyst with access to specialized tools to help you understand a csv that has been converted to a pandas dataframe.
        
    CRITICAL - READ CAREFULLY:
    You have access to tools that MUST be executed to get real results.
    YOU ARE NOT ALLOWED TO MAKE UP OR PREDICT RESULTS EVERY RESULT MUST BE READ THROUGH TOOL EXECUTION USING python_repl_ast.
    YOU MUST WAIT FOR THE ACTUAL TOOL EXECUTION OUTPUT.

    Available Tools:
    1. Basic Analysis:
        count_specific_actions()
        get_numeric_column_stats()
        find_most_common_actions()

    2. Advanced Analysis:
        consecutive_action_frequency()
        most_common_event_sequences()
        analyze_actions_after_distance()
        action_frequency_with_distance()
        multiple_actions_in_period()
        sequence_ending_with_action()

    HOW TO USE TOOLS:
    1. Write the tool call
    2. WAIT for actual execution using python_repl_ast
    3. Use ONLY the real output returned by the tool
    4. DO NOT make up or predict what the output might be

    These are the fields/columns available:
    {field_descriptions}

    Data Context:
    Total rows in dataset: {total_rows}
    Preview of data structure:
    {df_info}

    EXECUTION RULES:
    1. DO NOT create or predict outputs
    2. ONLY use actual tool execution results
    3. WAIT for tool execution before providing outputs
    4. If you don't see real output, say "Waiting for tool execution"
    5. NEVER fabricate tool outputs"""

    suffix = """EXECUTION PROCESS:
    1. Write tool code
    2. WAIT for execution
    3. Use ONLY the actual output
    4. NO fabricated results

    IF YOU DON'T SEE ACTUAL EXECUTION OUTPUT:
    Say "Waiting for tool execution..."

    CORRECT FORMAT:
    ```python
    result = consecutive_action_frequency(actions=["Sprint", "Deceleration"], max_time_between=2)
    print(result)
    ```
    [WAIT FOR ACTUAL EXECUTION OUTPUT]
    Then use only the real output for your conclusion."""

    formatted_prompt = system_prompt.format(
        total_rows=len(dataframe),
        df_info=str(dataframe.info()),
        field_descriptions="\n".join([f"- {k}: {v}" for k, v in field_descriptions.items()])
    )

    formatted_suffix = f"{suffix}"

    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
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