from langchain.document_loaders import PyPDFLoader , DirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document 
from typing import List 
import torch




# Extract text from PDF Files
def load_pdf_files (dataPath) : 
    loader = DirectoryLoader(dataPath , glob="*.pdf" , loader_cls=PyPDFLoader)
    docs = loader.load()
    return docs



def filter_Documents_metadata (docs:List[Document]) -> List[Document]:
    """
    Given a List of Documents objects , 
    return a new list of Document objects containing only 
    "source" in metadata and the original page_content
    """

    filtered_docs = []

    for doc in docs : 
        src = doc.metadata.get("source")
        filtered_docs.append(
            Document(
                page_content = doc.page_content ,
                metadata = {"source" :src}
            )
        )

    return filtered_docs


def text_split (filtered_docs) : 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000 , 
        chunk_overlap = 200 , 
    )

    text_chuncks = text_splitter.split_documents(filtered_docs)
    return text_chuncks 


def downloading_embeddings() : 
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name = model_name , 
        model_kwargs = {"device" : "cuda" if torch.cuda.is_available() else "cpu"}
    )
    return embeddings