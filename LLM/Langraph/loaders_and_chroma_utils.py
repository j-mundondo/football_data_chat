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
from typing import List
from langchain_core.documents import Document
import chromadb
# Initialize text splitter and embedding function
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Chroma with persistent client
client = chromadb.PersistentClient(path="./LLM/Langraph/chroma_db")
vectorstore = Chroma(
    client=client,
    embedding_function=embedding_function,
    collection_name="my_collection"
)

def load_and_split_document(file_path: str) -> List[Document]:
    """Load and split a document into chunks."""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        print("Successfully loaded PDF!")
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        print("Successfully loaded DOCX!")
    elif file_path.endswith(('.ppt', '.pptx')):
        loader = UnstructuredPowerPointLoader(file_path, mode="elements")
        print("Successfully loaded PowerPoint!")
    elif file_path.endswith(('.xls', '.xlsx')):
        loader = UnstructuredExcelLoader(file_path)
        print("Successfully loaded Excel!")
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
        print("Successfully loaded CSV!")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """Index a document to ChromaDB with error handling."""
    try:
        splits = load_and_split_document(file_path)
        
        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
            
        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False

def delete_doc_from_chroma(file_id: int):
    """Delete a document from ChromaDB by file_id."""
    try:
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False

def actually_index(path: str, file_id: int):
    """Main indexing function."""
    success = index_document_to_chroma(path, file_id)
    if success:
        print("Document successfully indexed to chroma")
    return success

# Example usage
file_path = r"C:\Users\j.mundondo\OneDrive - Statsports\Desktop\statsportsdoc\Projects\frequency_chat_PH\data\multi_session_hias\Research Paper.docx"
actually_index(file_path, file_id=1)