import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.helper import (
    download_hugging_face_embeddings,
    get_existing_ids,
    is_already_uploaded,
    load_pdf,
    make_id,
    text_split,
)

load_dotenv()
API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

if not API_KEY or not INDEX_NAME:
    raise ValueError("Add PINECONE_API_KEY and PINECONE_INDEX_NAME to your .env file")

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

if not chunks:
    raise SystemExit("No chunks found in data/. Stopping so the index isn't wiped.")

embeddings = download_hugging_face_embeddings()

# Give every chunk its ID
all_ids = [make_id(chunk) for chunk in chunks]

# Ask Pinecone which ones it already has (100 at a time)
existing_ids = get_existing_ids(index, all_ids)

# Keep only the chunks Pinecone doesn't have yet
new_chunks = []
new_ids = []

for chunk, chunk_id in zip(chunks, all_ids):
    if chunk_id not in existing_ids:
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
