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
        For basic questions that can be answered without a tool, just write the necessary python code and execute to get the answer. For example :
            "What's the most common action type?" does not require a tool. That can be done by work on the df
            "What's the average of distance?"
            "Whats the total distance" etc. do not require a tool. That can be done by work on the df
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
    4. If you don't see real output, say "Waiting for tool execution" as PythonAstREPLTool is called
    5. NEVER fabricate tool outputs
    
    CRITICAL : ALWAYS DOUBLE CHECK YOUR OUTPUT BEFORE PROVIDING AN ANSWER.
    """
    suffix = """EXECUTION PROCESS:
    1. Write tool code
    2. WAIT for execution of PythonAstREPLTool tool
    3. Use ONLY the actual output
    4. NO fabricated results

    IF YOU DON'T SEE ACTUAL EXECUTION OUTPUT:
    Say "Waiting for tool execution..."""

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
        prefix=formatted_prompt
        #,suffix=formatted_suffix
    )
    
    return agent