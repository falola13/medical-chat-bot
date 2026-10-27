import os

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)
from langchain_community.llms import CTransformers
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.helper import download_hugging_face_embeddings
from src.prompt import *

app = Flask(__name__)

load_dotenv()

API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

if not API_KEY or not INDEX_NAME:
    raise ValueError(
        "PINECONE_API_KEY is required. Generate one from pinecone website and add to .env"
    )

embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=API_KEY)
index_name = INDEX_NAME

# Loading the index
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)
retriever = docsearch.as_retriever(search_kwargs={"k": 2})

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={"max_new_tokens": 512, "temperature": 0.3, "context_length": 2048},
)


def join_chunks(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def stream_llama(prompt_value):
    prompt_text = prompt_value.to_string()
    yield from llm.client(prompt_text, stream=True)


chain = (
    {"context": retriever | join_chunks, "question": RunnablePassthrough()}
    | PROMPT
    | RunnableLambda(stream_llama)
)


@app.route("/")
def index():
    return render_template("chat.html")


# @app.route("/get", methods=["POST"])
# def chat():
#     question = request.form.get("msg", "").strip()

#     if not question:
#         return jsonify({"error": "Empty message"}), 400

#     print("Question:", question)
#     answer = chain.invoke(question)
#     print("Response:", answer)


#     return jsonify({"answer": answer})
@app.route("/stream", methods=["POST"])
def stream():
    question = request.form.get("msg", "").strip()

    if not question:
        return jsonify({"error": "Empty message"}), 400

    print("Question:", question)

    return Response(
        stream_with_context(chain.stream(question)),
        mimetype="text/plain",
    )


if __name__ == "__main__":
    app.run(debug=False)
