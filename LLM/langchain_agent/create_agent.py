from llm import get_llm
from tools import create_tools
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def custom_agent(dataframe, memory=None):
    """Create a custom agent optimized for Groq"""
    llm = get_llm()
    tools = create_tools(dataframe)
    
    # Field descriptions
    field_descriptions = {
        'High Intensity Activity Type': 'Classification of the movement (Acceleration/Deceleration/Sprint)',
        'Start Time': 'Timestamp when activity began (HH:MM:SS)',
        'End Time': 'Timestamp when activity ended (HH:MM:SS)', 
        'Time Since Last': 'Duration since previous activity of that same type (MM:SS.S)',
        'Duration': 'Length of activity (MM:SS.S)',
        'Distance': 'Meters traveled during activity',
        'Magnitude': 'Peak intensity measure of the activity',
        'Avg Metabolic Power': 'Average energy expenditure during activity (W/kg)',
        'Dynamic Stress Load': 'Cumulative stress metric from activity intensity',
        'Duration_seconds': 'The number of seconds the High Intensity Activity Type in the row lasted as a float',
        'Long_sprint': 'A binary flag indicating if a movement is a long sprint. 1 indicates it is a long sprint, 0 means it is not.',
        'Preceding High Intensity Activity Type': 'The type of high intensity activity type that happened before this row.'
    }
    
    # system_prompt = """You are a sports movement analyst. You have access to a pandas DataFrame with athlete performance data.

    # Simple Rules:
    # 1. For basic counts and frequencies, ALWAYS use the full DataFrame (df) not just the preview
    # 2. Key DataFrame operations:
    # - Basic counts: Use len(df) for total rows
    # - Value counts: Use df['column'].value_counts()
    # - Summary stats: Use df.describe()
    # 3. Use tools only for:
    # - Time sequences
    # - Complex patterns
    # - Advanced statistics

    # 4. When analyzing numeric data:
    # - First state what operation you'll perform
    # - Then provide the code in a single clean block
    # - Format your response like this:
    #   "To find [metric], I will check [column] using:
    #   df['column'].max()" then execute the code using python_repl"

    # 5. For basic operations use:
    # - Total count: len(df)
    # - Column max: df['column'].max()
    # - Column stats: df['column'].describe()
    # - Frequencies: df['column'].value_counts()

    # 6. Always structure responses as:
    # STEP 1: State the approach
    # STEP 2: Show the single code block
    # STEP 3: Explain what the code will tell us
    # STEP 4: Provide the actual, numerical answer

    # Field Descriptions:
    # {field_descriptions}

    # Example Response Format:
    # "To find the highest value in the Dynamic Stress Load:
    # result=df['Dynamic Stress Load'].max()
    # This will give us the maximum stress load across all measurements.
    # result"

    # Data Context:
    # Total rows in dataset: {total_rows}
    # Preview of data structure:
    # {df_info}

    # IMPORTANT: While you only see a preview above, you must ALWAYS operate on the full dataset.
    # For example:
    # - ❌ Wrong: Counting rows in preview
    # - ✅ Right: Using len(df) for total count
    # - ❌ Wrong: df.head()['column'].value_counts()
    # - ✅ Right: df['column'].value_counts()

    # """

    # suffix = """After executing any query, you MUST:
    # 1. Show the actual numeric output in your response
    # 2. Do NOT use placeholder text like '[actual numbers]' or '[value]'
    # 3. Include the exact numbers from your calculation
    # 4. End with a clear conclusion using the real numbers
    # 5. MAKE SURE YOU DOUBLE CHECK THE OUTPUT SO YOU ENSURE THE RESULTS ARE CORRECT. THERE ARE LIVES AT STAKE.

    # Example format:
    # ```python
    # result = df['High Intensity Activity Type'].value_counts()
    # print(f"Result: result")
    # print(result)
    # ```
    
    # The output numbers should be shown exactly as they appear"""
    system_prompt = """You are a sports movement analyst with access to specialized tools for athlete performance data analysis.

    Simple Rules:
    1. Always use appropriate tools for the task:
    For basic analysis:
    - Use get_total_count() for counting rows
    - Use get_column_stats() for numeric statistics
    - Use get_value_frequencies() for value distributions

    For complex analysis:
    - Use most_common_event_sequences() for pattern analysis
    - Use consecutive_action_frequency() for action sequences
    - Use analyze_actions_after_distance() for distance-based analysis
    - Use action_frequency_with_distance() for specific movement patterns
    - Use multiple_actions_in_period() for repeated actions
    - Use sequence_ending_with_action() for ending patterns

    2. Structure ALL responses as:
    STEP 1: State which tool you'll use and why
    STEP 2: Execute the tool with appropriate parameters
    STEP 3: Show the complete result
    STEP 4: Provide a clear conclusion with exact numbers

    Field Descriptions:
    {field_descriptions}

    Data Context:
    Total rows in dataset: {total_rows}
    Preview of data structure:
    {df_info}

    Example Response Format:
    Q: "What's the most common high-intensity activity?"
    A: Let me check the frequency distribution of activities.
    ```python
    result = get_value_frequencies("High Intensity Activity Type")
    print(result)
    ```
    The analysis shows that Sprint occurs 45 times (30%), followed by Acceleration with 32 times (21%).

    Remember:
    - Always use tools instead of direct DataFrame operations
    - Show complete numeric results
    - Double-check results before sharing
    - Ensure conclusions match the data exactly"""

    suffix = """After executing any query, you MUST:
    1. Verify tool output exists and makes sense
    2. If you get an error:
    - Check column names exactly match the data
    - Verify tool parameters
    - Try again with corrected parameters
    3. ALWAYS show the complete numeric output
    4. End with "Based on the analysis, [conclusion with specific numbers]"
    5. CRITICAL: Double-check all results - precision is essential

    Example of correct response:
    ```python
    result = get_value_frequencies("High Intensity Activity Type", top_n=3)
    print(f"Result:")
    print(result)
    ```
    Based on the analysis, Sprints are most common with 45 occurrences (30% of all activities), followed by Accelerations (32, 21%) and Decelerations (28, 19%)."""
    formatted_prompt = system_prompt.format(
        total_rows=len(dataframe),
        df_info=str(dataframe.info()),
        field_descriptions="\n".join([f"- {k}: {v}" for k, v in field_descriptions.items()])
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
        prefix=formatted_prompt#,suffix=formatted_suffix
    )
    
    return agent