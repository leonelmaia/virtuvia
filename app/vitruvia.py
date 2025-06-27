from utils import add_pdf_to_qdrant_index, OpenAI, QdrantClient
from qdrant_client.models import VectorParams, Distance
import sys
import csv
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

# Initialize clients
openai = OpenAI(
    api_key="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"
)
qdrant = QdrantClient(host="qdrant", port=6333)  # For docker compose execution
# qdrant = QdrantClient(host="localhost", port=6333)

collection_name = "docs"
embedding_dim = 1536

for _ in range(10):
    try:
        qdrant.get_collections()
        break
    except Exception as e:
        print("Waiting for Qdrant to be ready...")
        time.sleep(2)

logging.info(
    f"Existing collections: {[c.name for c in qdrant.get_collections().collections]}"
)

if collection_name not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
else:
    print("Collection 'docs' já existe, usando a existente.")
    points, _ = qdrant.scroll(
        collection_name="docs",
        limit=1000,
        with_payload=True,
    )

    sources = set()

    for point in points:
        source = point.payload.get("source")
        if source:
            sources.add(source)

    print("PDFs armazenados na coleção 'docs':")
    for src in sources:
        print(src)

doc_map = {}  # Or load with pickle
# doc_map = add_pdf_to_qdrant_index("documents/Leonel_CV.pdf", doc_map, openai=openai, qdrant=qdrant)
# doc_map = add_pdf_to_qdrant_index("documents/MINUTA_LUOS161123.pdf", doc_map, openai=openai, qdrant=qdrant)
# doc_map = add_pdf_to_qdrant_index("documents/minutadoprojetodelei_obraseedificacoes.pdf", doc_map, openai=openai, qdrant=qdrant)

df = pd.DataFrame(doc_map.items(), columns=["id", "text"])
df.to_csv("doc_map.csv", index=False, encoding="utf-8")
with open("doc_map.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "text"])  # header
    for doc_id, text in doc_map.items():
        writer.writerow([doc_id, text])
