import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    download_hugging_face_embeddings,
    is_already_uploaded,
    load_pdf,
    make_id,
    text_split,
)

load_dotenv()
API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

if not API_KEY or not INDEX_NAME:
    raise ValueError(
        "PINECONE_API_KEY is required. Generate one from pinecone website and add to .env"
    )

# Initializing the Pinecone
pc = Pinecone(api_key=API_KEY)

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )


index = pc.Index(INDEX_NAME)

documents = load_pdf("data/")
chunks = text_split(documents)
embeddings = download_hugging_face_embeddings()

new_chunks = []
new_ids = []

for chunk in chunks:
    chunk_id = make_id(chunk)
    if not is_already_uploaded(index, chunk_id):
        new_chunks.append(chunk)
        new_ids.append(chunk_id)

print(f"Total chunks: {len(chunks)}")
print(f"New chunks to upload: {len(new_chunks)}")

if len(new_chunks) == 0:
    print("Nothing new. Index is up to date.")
else:
    PineconeVectorStore.from_documents(
        documents=new_chunks, embedding=embeddings, index_name=INDEX_NAME, ids=new_ids
    )
    print("Upload done.")

# Deleted old chunks from pinecone
current_ids = {make_id(chunk) for chunk in chunks}

deleted_count = 0

for batch_of_ids in index.list():
    old_ids = [i for i in batch_of_ids if i not in current_ids]
    if old_ids:
        index.delete(ids=old_ids)
        deleted_count += len(old_ids)

if deleted_count > 0:
    print(f"Removed {deleted_count} old chunks.")

# docsearch = PineconeVectorStore.from_existing_index(INDEX_NAME, embedding=embeddings)
