import pandas as pd
from custom_agents_LG import agent_executor
from llm import get_llama_3dot3_70b_versatile,get_llama_3dot1_8b_instant,get_70b_8192
llm = get_70b_8192()


# Import necessary libraries
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader, 
    UnstructuredPowerPointLoader, 
    CSVLoader, 
    UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
import chromadb

# Initialize components
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Chroma with persistent client
client = chromadb.PersistentClient(path=".\LLM\Langraph\chroma_db")# C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\LLM\Langraph\chroma_db
vectorstore = Chroma(
    client=client,
    embedding_function=embedding_function,
    collection_name="my_collection"
)

# Define State class
class State(TypedDict):
    question: str
    context: List[Document]
    player_metrics: str
    answer: str
    player_name: str

# Document loading functions
def load_and_split_document(file_path: str) -> List[Document]:
    """Load and split a document into chunks."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith(('.ppt', '.pptx')):
        loader = UnstructuredPowerPointLoader(file_path, mode="elements")
    elif file_path.endswith(('.xls', '.xlsx')):
        loader = UnstructuredExcelLoader(file_path)
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import os

# Initialize text splitter and embedding function
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the ChromaDB path
CHROMA_PATH = "./LLM/Langraph/chroma_db"

def get_vectorstore():
    """Get or create vectorstore with fresh client connection."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return Chroma(
        client=client,
        embedding_function=embedding_function,
        collection_name="my_collection"
    )

def verify_document_loading(file_path: str):
    """Load, index, and verify document loading with detailed debugging."""
    print("\nSTEP 1: Loading document...")
    try:
        splits = load_and_split_document(file_path)
        print(f"Successfully split document into {len(splits)} chunks")
        print("\nFirst chunk preview:")
        if splits:
            print(splits[0].page_content[:200])
        
        print("\nSTEP 2: Resetting collection...")
        # Get fresh vectorstore connection
        vectorstore = get_vectorstore()
        
        # Get and delete existing documents
        existing_ids = vectorstore._collection.get()['ids']
        if existing_ids:
            vectorstore._collection.delete(ids=existing_ids)
        print("Collection reset complete")
        
        print("\nSTEP 3: Indexing document chunks...")
        for i, split in enumerate(splits):
            split.metadata['file_id'] = 1
            split.metadata['chunk_id'] = i
        
        vectorstore.add_documents(splits)
        print(f"Added {len(splits)} chunks to vectorstore")
        return True
    except Exception as e:
        print(f"ERROR during document processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Graph nodes
def extract_player_name(question: str) -> str:
    """Extract player name from the question using the LLM."""
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

def setup_initial_state(question: str) -> State:
    """Setup initial state without metrics - moved to separate node."""
    return {
        "question": question,
        "context": [],
        "player_metrics": "",
        "answer": "",
        "player_name": ""
    }

def extract_name_node(state: State):
    """Node for extracting player name."""
    player_name = extract_player_name(state["question"])
    return {"player_name": player_name}

def get_metrics_node(state: State):
    """Node for getting player metrics."""
    if not state["player_name"]:
        return {"player_metrics": "No specific player mentioned in query"}
    
    metrics = get_player_metrics(state["player_name"])
    return {"player_metrics": metrics}

def retrieve_node(state: State):
    """Node for retrieving relevant documents."""
    documents = []
    
    if state["player_metrics"] != "No specific player mentioned in query":
        metrics_doc = Document(
            page_content=f"Current Player Metrics for {state['player_name']}:\n{state['player_metrics']}",
            metadata={"source": "player_metrics"}
        )
        documents.append(metrics_doc)
    
    # Get research documents with specific queries
    research_queries = [
        f"injury risk thresholds for {state['player_name']}'s activity pattern",
        "critical thresholds for Dynamic Stress Load DSL",
        "metabolic power thresholds for injury risk",
        "recovery indicators and patterns research",
        "acceleration deceleration ratio research findings"
    ]
    
    for query in research_queries:
        retrieved_docs = vectorstore.similarity_search(
            query,
            k=2
        )
        documents.extend(retrieved_docs)
    
    return {"context": documents}

def generate_node(state: State):
    """Generate analysis with explicit reference to research data."""
    if state["player_metrics"] == "No specific player mentioned in query":
        return {"answer": "To analyze a player's injury risks, please specify a player name in your query."}
    
    # Separate metrics and research documents
    metrics_doc = None
    research_docs = []
    
    for doc in state["context"]:
        if doc.metadata.get("source") == "player_metrics":
            metrics_doc = doc
        else:
            research_docs.append(doc)
    
    # Format research findings
    research_findings = "\n\n".join([
        f"RESEARCH FINDING {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(research_docs)
    ])
    
    analysis_prompt = """
    Using ONLY the provided research data and player metrics, conduct an injury risk analysis.
    
    PLAYER METRICS:
    {metrics}
    
    RESEARCH FINDINGS:
    {research}
    
    Analyze the following aspects, citing ONLY the provided research:
    1. Compare the player's metrics to the research thresholds
    2. Identify specific risk factors supported by the research
    3. Make recommendations based on the research findings
    
    Format your response with clear sections and evidence from the provided research.
    """
    
    messages = PromptTemplate.from_template(analysis_prompt).invoke({
        "metrics": metrics_doc.page_content if metrics_doc else "No metrics available",
        "research": research_findings
    })
    
    response = llm.invoke(messages)
    return {"answer": response.content}

def build_streaming_graph():
    """Build graph with streaming support."""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("extract_name", extract_name_node)
    workflow.add_node("get_metrics", get_metrics_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    # Add edges
    workflow.add_edge("extract_name", "get_metrics")
    workflow.add_edge("get_metrics", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Set entry point
    workflow.set_entry_point("extract_name")
    
    return workflow.compile()

def analyze_player_risk_streaming(question: str):
    """Stream the analysis process."""
    graph = build_streaming_graph()
    initial_state = setup_initial_state(question)
    
    print("Starting Analysis Stream:")
    print("-" * 50)
    
    for step in graph.stream(initial_state, stream_mode="updates"):
        print("Step Update:")
        for key, value in step.items():
            if key == "context":
                print(f"{key}: {len(value)} documents retrieved")
            else:
                print(f"{key}: {value}")
        print("-" * 50)
    
    return graph.invoke(initial_state)

# Main execution
if __name__ == "__main__":
    # First load and verify research document
    file_path = r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\Research Paper.docx"
    success = verify_document_loading(file_path)
    
    if success:
        # Test queries
        test_queries = [
            "What are Lee's injury risks based on his recent activity pattern?",
            "Analyze injury risks for Hawk",
            "What are the injury risks for this player's recent pattern?"
        ]
        
        for query in test_queries:
            print(f"\nAnalyzing query: {query}")
            print("-" * 50)
            response = analyze_player_risk_streaming(query)
            print("Analysis Results:")
            print(response["answer"])
            print("-" * 50)