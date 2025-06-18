import os
import logging
import shutil
from glob import glob
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from openai import OpenAI
from qdrant_client.http import models as rest_models
import numpy as np
import fitz
from sklearn.preprocessing import normalize


def preprocess_chunk(text):
    # Your custom cleaning steps here
    return text.strip()


def move_pdf_files(file):
    save_dir = "documents"
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(file)
    save_path = os.path.join(save_dir, file_name)
    shutil.move(file.name, save_path)

    return f"Arquivo salvo em {save_path}"

def add_pdf_to_qdrant_index(
    pdf_path,
    doc_map,
    openai,
    qdrant,
    chunk_size=1024,
    overlap=128,
    collection_name="docs",
):
    pdf_path = f"documents/{pdf_path}"
    doc = fitz.open(pdf_path)

    chunks = []
    metadata_list = []

    for page_number, page in enumerate(doc):
        page_text = page.get_text().strip()
        start = 0

        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            chunk = preprocess_chunk(page_text[start:end])
            chunk = chunk.strip()

            if chunk:  # Skip empty chunks
                chunks.append(chunk)
                metadata_list.append({
                    "source": pdf_path,
                    "page": page_number + 1,  # 1-based page number
                })

            start += chunk_size - overlap

    # Generate embeddings
    embeddings = []
    for chunk in chunks:
        resp = openai.embeddings.create(input=chunk, model="text-embedding-ada-002")
        embeddings.append(resp.data[0].embedding)

    embeddings = normalize(np.array(embeddings).astype("float32"), axis=1)

    current_max_id = max(doc_map.keys()) if doc_map else -1
    points = []
    for i, (chunk, emb, meta) in enumerate(zip(chunks, embeddings, metadata_list)):
        idx = current_max_id + i + 1
        points.append({
            "id": idx,
            "vector": emb.tolist(),
            "payload": {
                "content": chunk,
                "source": meta["source"],
                "page": meta["page"],
            }
        })
        doc_map[idx] = {
            "content": chunk,
            "source": meta["source"],
            "page": meta["page"],
        }

    qdrant.upsert(collection_name=collection_name, points=points)

    print(f"✅ Added {len(chunks)} chunks from {pdf_path}")
    return doc_map

def remove_temp_gradio_file(pdf_name: str):
    path = os.path.join("/app/documents", pdf_name)
    if os.path.exists(path):
        os.remove(path)
        logging.info(f"✅ PDF '{pdf_name}' removido com sucesso.")
        return 
    else:
        logging.info(f"❌ PDF '{pdf_name}' não encontrado.")
        return


def delete_pdf_from_qdrant(pdf_name, qdrant,doc_map, collection_name="docs"):
    pdf_path = f"documents/{pdf_name}"
    filter_condition = rest_models.Filter(
        must=[
            rest_models.FieldCondition(
                key="source",
                match=rest_models.MatchValue(value=pdf_path)
            )
        ]
    )

    points_selector = rest_models.FilterSelector(filter=filter_condition)

    qdrant.delete(
        collection_name=collection_name,
        points_selector=points_selector
    )

    keys_to_delete = [k for k, v in doc_map.items() if os.path.basename(v['source']) == pdf_path]
    for k in keys_to_delete:
        del doc_map[k]
    logging.info(f"PDF '{pdf_name}' removido do Qdrant e do doc_map")
    return 



def rebuild_doc_map(qdrant, collection_name="docs"):
    doc_map = {}
    response = qdrant.scroll(
        collection_name=collection_name,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    for point in response[0]:
        point_id = point.id
        payload = point.payload
        doc_map[point_id] = {
            "content": payload.get("content"),
            "source": payload.get("source"),
            "page": payload.get("page"),
        }
    return doc_map

def get_context_from_qdrant(query, openai, qdrant, k=5):
    embedding_resp = openai.embeddings.create(input=query, model="text-embedding-ada-002")
    query_vector = embedding_resp.data[0].embedding

    results = qdrant.search(
        collection_name="docs",
        query_vector=query_vector,
        limit=k,
        with_payload=True
    )
    context_chunks  = []
    for r in results:
        text = r.payload.get("content", "")
        source = r.payload.get("source", "Unknown. Please check this information")
        page = r.payload.get("page", "Please check this information")
        context_chunks.append((source, page, text))

    return context_chunks 

def build_messages(context, chat_history, user_input):
    system_prompt = (
        "Você é um assistente prestativo especializado em arquitetura e design."
        "Use o contexto dos documentos fornecidos para responder às perguntas do usuário da forma mais precisa possível."
        "Se a pergunta do usuário estiver pouco clara ou ambígua, faça uma ou duas perguntas curtas de esclarecimento antes de responder."
        "Se o contexto dos documentos não for suficiente para responder completamente à pergunta, combine-o com seu conhecimento geral."
        "Se sua resposta contiver informações que não estejam nos documentos, identifique claramente como “Baseado em conhecimento geral”."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append({
            "role": "system",
            "content": f"Relevant document context (extracted from local files):\n{context}"
        })

    for turn in chat_history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    messages.append({"role": "user", "content": user_input})

    return messages


def get_embedding(text, openai):
    resp = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    return np.array(resp.data[0].embedding).astype("float32")

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def filter_relevant_sources(answer, chunks_with_sources, openai, threshold=0.85):
    answer_emb = get_embedding(answer, openai)
    relevant_sources = set()

    for source, page, chunk_text in chunks_with_sources:
        chunk_emb = get_embedding(chunk_text, openai)
        sim = cosine_similarity(answer_emb, chunk_emb)
        logging.info(f"Answer: {answer}")
        logging.info(f"Chunk: {chunk_text}")
        logging.info(f"Similarity: {sim}")
        if sim >= threshold:
            relevant_sources.add((source, page))
    logging.info(f"Relevant sources: {relevant_sources}")
    return relevant_sources
