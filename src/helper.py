import hashlib
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone.db_data.index import Index


# Extract data from pdf
def load_pdf(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents


# Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


# download the embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def make_id(chunk):
    text = chunk.page_content
    return hashlib.md5(text.encode()).hexdigest()


def is_already_uploaded(index: Index, chunk_id):
    result = index.fetch(ids=[chunk_id])
    return chunk_id in result.vectors
