from flask import Flask , render_template , jsonify , request 
from src.helper import downloading_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompts import *
import os


pineConeApi = os.getenv("pineConeApi_1")
openaiApiKey = os.getenv("openApiKey")
base_url = os.getenv("openApiBaseUrl")
os.environ["PINECONE_API_KEY"] = pineConeApi
os.environ["PINECONE_ENV"] = "us-east-1" 
os.environ['OPEN_API_KEY'] = openaiApiKey



index_name = "medical-chatbot"
embeddings = downloading_embeddings()
docSearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings , 
    index_name = index_name 
)

retriever = docSearch.as_retriever(search_type = "similarity" , search_kwargs = {"k" : 5})
chatModel = ChatOpenAI (
    model="openai/gpt-oss-20b:free",
    base_url=base_url,
    api_key=openaiApiKey
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , sys_prompt) , 
        ("human" , "{input}")
    ]
)


question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever , question_answer_chain)


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        # If frontend sends JSON (recommended)
        data = request.get_json()
        user_msg = data.get("msg", "")
    else:
        # If frontend uses ?msg=...
        user_msg = request.args.get("msg", "")

    # RAG invocation
    result = rag_chain.invoke({"input": user_msg})

    return jsonify({"answer": result["answer"]})



if __name__ == '__main__': 
    app.run(host="0.0.0.0", port=8080, debug=True)









