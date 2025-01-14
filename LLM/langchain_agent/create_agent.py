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
    system_prompt = """You are a sports movement analyst with access to specialized tools that MUST be used for all data analysis. Direct DataFrame operations are NOT allowed.

    CRITICAL: You must EXECUTE the tools to get actual results. Never return placeholder text or say the answer will be provided later.

    Available Tools (YOU MUST USE THESE):
    1. For Basic Analysis:
        - count_specific_actions(): For counting action types (e.g., "How many sprints?", "Count all decelerations")
        - get_numeric_column_stats(): For statistics on numeric columns (e.g., "What's the average sprint distance?")
        - find_most_common_actions(): For frequency analysis (e.g., "What are the top 3 most frequent activities?")

    2. For Complex Analysis:
        - most_common_event_sequences(): For identifying common patterns (e.g., "What actions typically follow sprints?")
        - consecutive_action_frequency(): For analyzing action sequences (e.g., "How often do sprints follow accelerations?")
        - analyze_actions_after_distance(): For distance-based patterns (e.g., "What follows sprints over 20m?")
        - action_frequency_with_distance(): For analyzing movement magnitudes (e.g., "How often are sprints longer than 30m?")
        - multiple_actions_in_period(): For repeated actions (e.g., "How often do players sprint twice in 10 seconds?")
        - sequence_ending_with_action(): For analyzing action chains (e.g., "What leads to decelerations?")

    ALWAYS answer questions using this format:
    1. "I'll use [specific tool name] to analyze this"
    2. Execute the tool with appropriate parameters
    3. Show the complete numeric output
    4. Provide conclusion with exact numbers

    Example Correct Answer:
    Q: "How many sprints are there?"
    A: I'll use count_specific_actions to find the number of sprints.
    Tool execution:
    ```python
    result = count_specific_actions(action_type="Sprint")
    print(result)
    ```
    Tool output: {'action_type': 'Sprint', 'count': 45, 'percentage': 30.2, 'total_actions': 149}
    Based on the data, there are exactly 45 sprints, representing 30.2% of all actions.

    Field Descriptions:
    {field_descriptions}

    Data Context:
    Total rows in dataset: {total_rows}
    Preview of data structure:
    {df_info}

    IMPORTANT: 
    - NEVER use direct DataFrame operations (df[])
    - ALWAYS use and execute the provided tools
    - ALWAYS show actual numeric results
    - NEVER say "answer will be provided" or use placeholders"""

    suffix = """EXECUTION REQUIREMENTS:
    1. You MUST use one of the provided tools
    2. You MUST execute the tool and show its output
    3. You MUST include actual numbers in your conclusion
    4. You MUST verify the results are correct

    NEVER say:
    ❌ "The answer will be provided"
    ❌ "[Actual numbers will appear here]"
    ❌ "Result: [value]"

    INSTEAD, always show:
    ✅ Actual tool execution
    ✅ Complete numeric output
    ✅ Specific conclusion with real numbers

    Example:
    Q: "What's the most common action?"
    ```python
    result = find_most_common_actions(top_n=3)
    print(result)
    ```
    Tool output: {
        'top_actions': {
            'Sprint': {'count': 45, 'percentage': 30.2},
            'Acceleration': {'count': 32, 'percentage': 21.5},
            'Deceleration': {'count': 28, 'percentage': 18.8}
        },
        'total_actions': 149,
        'unique_actions': 3
    }
    Based on this data, Sprints are most common with exactly 45 occurrences (30.2% of all actions), followed by Accelerations (32 occurrences, 21.5%) and Decelerations (28 occurrences, 18.8%)."""
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