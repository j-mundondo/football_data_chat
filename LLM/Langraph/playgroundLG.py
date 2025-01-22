import pandas as pd
from custom_agents_LG import load_session_data, agent_executor
from llm import get_llama_3dot3_70b_versatile,get_llama_3dot1_8b_instant,get_70b_8192
from loaders_and_chroma_utils import vectorstore
from langchain_core.prompts import PromptTemplate

llm = get_70b_8192()
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langgraph.graph import START, StateGraph
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List
from langchain_core.documents import Document

# # Modified prompt template that explicitly references player metrics
template = """Using the following research data and current player metrics, analyze the potential injury risks and performance insights.

Research Context:
{context}

Current Player Metrics:
{player_metrics}

Analysis Request: {question}

Based on both the research findings and the current metrics, provide:
1. Specific risk indicators identified in the current metrics
2. How these metrics compare to research thresholds
3. Evidence-based recommendations for load management

First show the players player metrics before going into the analysis if you cannot find the metrics tell me.

Sports Science Analysis:"""

custom_rag_prompt = PromptTemplate.from_template(template)

class State(TypedDict):
    question: str
    context: List[Document]
    player_metrics: str
    answer: str
    player_name: str  # Added to track player name

def extract_player_name(question: str) -> str:
    """
    Extract player name from the question using the LLM.
    This is more robust than regex as it can handle various phrasings.
    """
    name_extraction_prompt = """
    Extract the player name(s) from the following question. 
    If no specific player is mentioned, return None.
    Only return the name(s) without any additional text.
    
    Question: {question}
    """
    
    prompt = PromptTemplate.from_template(name_extraction_prompt)
    messages = prompt.invoke({"question": question})
    response = llm.invoke(messages).content.strip()
    
    return response if response.lower() != "none" else None

def get_player_metrics(player_name: str) -> dict:
    """Get metrics for a specific player."""
    return agent_executor.invoke({
        "input": f"Get {player_name}'s metrics as a dictionary format"
    })["output"]

def setup_state(question: str) -> State:
    """Initialize state with player name and metrics."""
    player_name = extract_player_name(question)
    
    if not player_name:
        return {
            "question": question,
            "context": [],
            "player_metrics": "No specific player mentioned in query",
            "answer": "",
            "player_name": ""
        }
    
    metrics = get_player_metrics(player_name)
    
    return {
        "question": question,
        "context": [],
        "player_metrics": metrics,
        "answer": "",
        "player_name": player_name
    }

def retrieve(state: State):
    """Retrieve relevant documents and combine with player metrics."""
    retrieved_docs = vectorstore.similarity_search(state["question"])
    
    if state["player_metrics"] != "No specific player mentioned in query":
        metrics_doc = Document(
            page_content=f"Current Player Metrics for {state['player_name']}:\n{state['player_metrics']}",
            metadata={"source": "player_metrics"}
        )
        retrieved_docs.insert(0, metrics_doc)
    
    return {"context": retrieved_docs}

def generate(state: State):
    """Generate response using context and metrics."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    if state["player_metrics"] == "No specific player mentioned in query":
        analysis_prompt = """
        To analyze a player's injury risks, please specify a player name in your query.
        Available players can be queried from the system.
        """
        return {"answer": analysis_prompt}
    
    messages = custom_rag_prompt.invoke({
        "question": state["question"],
        "context": docs_content,
        "player_metrics": state["player_metrics"],
        "player_name": state["player_name"]
    })
    
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build and run the graph
def analyze_player_risk(question: str):
    """Main function to analyze player risk based on a question."""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Compile graph
    graph = workflow.compile()
    
    # Initialize state with player info
    initial_state = setup_state(question)
    
    # Run analysis
    response = graph.invoke(initial_state)
    
    return response

# Example usage
if __name__ == "__main__":
    # Test with different queries
    queries = [
        "What are Hawk's injury risks based on his recent activity pattern?"
        # "Analyze the injury risks for Hawk and Ben",
        # "What are the injury risks based on recent activity patterns?",  # No specific player
    ]
    
    for query in queries:
        print(f"\nAnalyzing query: {query}")
        print("-" * 50)
        response = analyze_player_risk(query)
        print("Analysis Results:")
        print(response["answer"])
        print("-" * 50)