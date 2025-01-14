from llm import get_llm
from tools import create_tools
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
llm = get_llm()

from llm import get_llm
from tools import create_tools
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

prefix = '''You are a friendly assistant named PEXI who works for a sports data company.
            Your job is to answer questions regarding the dataset using relevant and 
            respond in a friendly manner. Do not general questions outside the realm of
            this dataset e.g. general knowledge questions. Also, only use tools when needed
            some simple questions do not need tools so only use them when needed. You have
            acess to the following tools:
            '''
suffix = ''

# Create a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prefix),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}"),
    ]
)

def custom_agent(dataframe, memory=None):
    """
    Create a custom agent with optional memory integration
    Args:
        dataframe: pandas DataFrame to analyze
        memory: optional memory instance for conversation history
    """
    llm = get_llm()
    tools = create_tools(dataframe)
    
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        prefix=prompt,
        extra_tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        agent_type="tool-calling",
        memory=memory  # Pass memory directly to agent creation
    )
    
    return agent