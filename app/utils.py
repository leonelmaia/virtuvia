import os
import logging
import random
import shutil
import uuid
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.models import PointStruct, VectorParams, Distance
import numpy as np
import fitz
import requests
from sklearn.preprocessing import normalize
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


def preprocess_chunk(text):
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
                metadata_list.append(
                    {
                        "source": pdf_path,
                        "page": page_number + 1,
                    }
                )

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
        points.append(
            {
                "id": idx,
                "vector": emb.tolist(),
                "payload": {
                    "content": chunk,
                    "source": meta["source"],
                    "page": meta["page"],
                },
            }
        )
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


def delete_pdf_from_qdrant(pdf_name, qdrant, doc_map, collection_name="docs"):
    pdf_path = f"documents/{pdf_name}"
    filter_condition = rest_models.Filter(
        must=[
            rest_models.FieldCondition(
                key="source", match=rest_models.MatchValue(value=pdf_path)
            )
        ]
    )

    points_selector = rest_models.FilterSelector(filter=filter_condition)

    qdrant.delete(collection_name=collection_name, points_selector=points_selector)

    keys_to_delete = [
        k for k, v in doc_map.items() if os.path.basename(v["source"]) == pdf_path
    ]
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
        with_vectors=False,
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
    embedding_resp = openai.embeddings.create(
        input=query, model="text-embedding-ada-002"
    )
    query_vector = embedding_resp.data[0].embedding
    collections = qdrant.get_collections().collections
    all_results = []

    for col in collections:
        col_name = col.name
        try:
            results = qdrant.search(
                collection_name=col_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
            )
            for r in results:
                text = r.payload.get("content", "") or r.payload.get("conteudo", "")
                source = r.payload.get("source", col_name)
                page = r.payload.get("page", "N/A")
                all_results.append(
                    {
                        "collection": col_name,
                        "score": r.score,
                        "source": source,
                        "page": page,
                        "content": text,
                    }
                )
            logging.info(f"{col_name} => All_results \n\n '{all_results}'")
        except Exception as e:
            print(f"Erro ao buscar na coleção '{col_name}': {e}")
            continue

    all_results.sort(key=lambda x: x["score"], reverse=True)
    top_k = all_results[:k]
    context_chunks = [(r["source"], r["page"], r["content"]) for r in top_k]
    return context_chunks


def get_documents_from_lexml(keyword, limite=1):
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--width=1920")
    options.add_argument("--height=1080")
    options.set_preference(
        "general.useragent.override",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    )

    logging.info("Iniciando driver Firefox...")
    driver = webdriver.Remote(
        command_executor="http://selenium:4444/wd/hub", options=options
    )

    data = []
    try:
        driver.get("https://www.lexml.gov.br/")
        time.sleep(random.uniform(2, 4))

        search_box = WebDriverWait(driver, 20).until(
            EC.visibility_of_element_located((By.NAME, "keyword"))
        )
        search_box.send_keys(keyword)
        driver.find_element(By.CSS_SELECTOR, "input[type='submit']").click()

        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.XPATH, "//a[starts-with(@href, '/urn/')]")
            )
        )
        results = driver.find_elements(By.XPATH, "//a[starts-with(@href, '/urn/')]")

        for link in results[:limite]:
            title = link.text.strip()
            relative_url = link.get_attribute("href")
            full_url = (
                f"https://www.lexml.gov.br{relative_url}"
                if relative_url.startswith("/")
                else relative_url
            )
            logging.info(f"Abrindo {full_url}")
            driver.get(full_url)
            time.sleep(random.uniform(2, 4))

            pub_href = None
            compilado_href = None

            try:
                WebDriverWait(driver, 10).until(
                    lambda d: d.execute_script("return document.readyState")
                    == "complete"
                )

                pub_link = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//a[contains(@href, 'planalto.gov.br')]")
                    )
                )
                pub_href = pub_link.get_attribute("href")
                time.sleep(random.uniform(2, 4))
                logging.info(f"pub_href => {pub_href}")

                driver.get(pub_href)
                compilado_link = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//a[contains(@href, 'compilado.htm')]")
                    )
                )
                compilado_href = compilado_link.get_attribute("href")
                logging.info(f"Link do texto compilado encontrado: {compilado_href}")

            except Exception as e:
                logging.warning(f"Nenhum link de publicação encontrado: {e}")

            data.append(
                {
                    "title": title,
                    "link_resultado": full_url,
                    "link_publicacao_original": pub_href,
                    "link_publicacao_compilado": compilado_href,
                }
            )

    finally:
        driver.quit()
        logging.info("Driver encerrado.")

    logging.info(f"DADOS => {data}")
    return data


def extract_law_text(link):
    try:
        r = requests.get(link)
        soup = BeautifulSoup(r.text, "html.parser")
        content = soup.get_text(separator="\n")
        logging.info(f"Content '{content}'")
        return content
    except Exception:
        return ""


def process_documents_lexml(keyword, openai, qdrant):
    chunk_size = 1024
    overlap = 128
    logging.info(f"Getting context...")
    try:
        docs = get_documents_from_lexml(keyword)
    except Exception as e:
        logging.warning(f"Erro ao obter documentos: {e}")
    logging.info("Retornou de get_documents_from_lexml")
    logging.info(f"Docs => {docs}")
    if "legislacao_brasil" not in [
        c.name for c in qdrant.get_collections().collections
    ]:
        qdrant.create_collection(
            collection_name="legislacao_brasil",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    else:
        logging.info(f"Collection 'legislacao_brasil' já existe, usando a existente.")
        qdrant.scroll(
            collection_name="legislacao_brasil",
            limit=1000,
            with_payload=True,
        )

    for doc in docs:
        texto = extract_law_text(doc["link_publicacao_compilado"])
        if not texto:
            continue

        chunks = []
        metadata_list = []

        start = 0
        while start < len(texto):
            end = min(start + chunk_size, len(texto))
            chunk = texto[start:end].strip()

            if chunk:
                chunks.append(chunk)
                metadata_list.append(
                    {
                        "Title": doc["titulo"],
                        "source": doc["link_publicacao_compilado"],
                        "page": "N/A",
                        "content": chunk,
                    }
                )

            start += chunk_size - overlap

        # Gerar embeddings com OpenAI
        embeddings = []
        for chunk in chunks:
            try:
                resp = openai.embeddings.create(
                    input=chunk, model="text-embedding-ada-002"
                )
                embeddings.append(resp.data[0].embedding)
            except Exception as e:
                logging.info(f"Erro ao gerar embedding: {e}")
                continue

        # Montar e enviar os pontos para o Qdrant
        points = [
            PointStruct(
                id=str(uuid.uuid4()), vector=embeddings[i], payload=metadata_list[i]
            )
            for i in range(len(embeddings))
        ]

        if points:
            qdrant.upsert(collection_name="legislacao_brasil", points=points)
        logging.info(f"Legislacao adicioanda ao embedding com sucesso")


def extract_keyword(msg, openai):
    prompt = f"""
    Extraia a principal palavra ou expressão-chave jurídica desta pergunta para ser usada em uma busca legal:
    
    "{msg}"
    
    Responda apenas com a palavra ou expressão, sem explicações.
    """
    resposta = openai.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.2
    )
    return resposta.choices[0].message.content.strip()


def build_messages(context, chat_history, user_input):
    system_prompt = (
        "Responda como se você fosse o melhor advogado do Brasil utilizando termos juridicos."
        "Cite qual a lei e quais os artigos."
        "Use o contexto dos documentos fornecidos para responder às perguntas do usuário da forma mais precisa possível."
        "Se a pergunta do usuário estiver pouco clara ou ambígua, faça uma ou duas perguntas curtas de esclarecimento antes de responder."
        "Se o contexto dos documentos não for suficiente para responder completamente à pergunta, combine-o com seu conhecimento geral."
        "Se sua resposta contiver informações que não estejam nos documentos, identifique claramente como “Baseado em conhecimento geral”."
    )

    messages = [{"role": "system", "content": system_prompt}]

    if context:
        messages.append(
            {
                "role": "system",
                "content": f"Relevant document context (extracted from local files):\n{context}",
            }
        )

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
