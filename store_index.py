import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files , filter_Documents_metadata , text_split , downloading_embeddings


load_dotenv()

pineConeApi = os.getenv("pineConeApi_1")
openaiApiKey = os.getenv("openApiKey")
base_url = os.getenv("openApiBaseUrl")

os.environ["PINECONE_API_KEY"] = pineConeApi
os.environ['OPEN_API_KEY'] = openaiApiKey
os.environ["PINECONE_ENV"] = "us-east-1" 




pc = Pinecone(api_key=pineConeApi)
index_name = "medical-chatbot"

if not pc.has_index(index_name) : 
    pc.create_index(
        name=index_name , 
        dimension=384 , 
        metric="cosine" , 
        spec=ServerlessSpec(cloud="aws" , region="us-east-1")
    )

index = pc.Index(index_name)


extractedData = load_pdf_files("data")
filtered_docs = filter_Documents_metadata(extractedData)
text_chunks = text_split(filtered_docs)
embeddings = downloading_embeddings()


docSearch = PineconeVectorStore.from_documents(
    documents=text_chunks , 
    embedding=embeddings , 
    index_name = index_name 
)