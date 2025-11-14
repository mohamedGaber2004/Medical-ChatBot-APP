# app.py
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

# LangChain imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory

# Custom helpers
from memory_store import get_memory
from src.helper import downloading_embeddings
from src.prompts import sys_prompt

# Load environment variables
load_dotenv()
pinecone_api = os.getenv("pineConeApi_1")
openai_api_key = os.getenv("openApiKey")
base_url = os.getenv("openApiBaseUrl")

# Set environment variables for libraries
os.environ["PINECONE_API_KEY"] = pinecone_api
os.environ["PINECONE_ENV"] = "us-east-1"
os.environ["OPEN_API_KEY"] = openai_api_key

# --- Embeddings and Pinecone setup ---
embeddings = downloading_embeddings()
index_name = "medical-chatbot"
docSearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)
retriever = docSearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# --- Chat model setup ---
chat_model = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url=base_url,
    api_key=openai_api_key
)

# --- Prompt template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    ("human", "{input}")
])

# --- Chain setup ---
question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Conversation chain with memory ---
conversation_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: get_memory(session_id),  # Returns memory object
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer"
)

# --- Flask app ---
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    # Get user message
    if request.is_json:
        data = request.get_json()
        user_msg = data.get("msg", "")
    else:
        user_msg = request.form.get("msg", "")

    # Get session ID for memory
    session_id = request.cookies.get("session_id", "default_session")

    # Invoke chain with memory
    output = conversation_chain.invoke(
        {"input": user_msg},
        {"configurable": {"session_id": session_id}}
    )
    
    answer = output.get("answer") if isinstance(output, dict) else str(output)
    return jsonify({"status":"success","answer": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
