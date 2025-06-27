# Virtuvia

Plataforma de RAG e Raciocínio Jurídico com Qdrant, Selenium e APIs públicas.

## 🧩 Visão geral

Este projeto objetiva construir um pipeline completo para:

- Extrair e buscar documentos jurídicos (Lei, Jurisprudência) via LexML  
- Processar e indexar conteúdo em embeddings com OpenAI + Qdrant  
- Integrar recuperação de contexto (RAG) em um chatbot  
- Orquestrar automações com Selenium e Docker Compose

Ideal para pesquisa jurídica, triagem documental e preparação para agentes conversacionais com fundamento técnico + legal.

## 🚀 Funcionalidades

- Scraping com Selenium (ou alternativa como undetected‑chromedriver)  
- Suporte a múltiplos navegadores headless  
- Conversão de PDF em texto, chunking, embeddings  
- Armazenamento e busca vetorial com Qdrant  
- API e UI interativa (FastAPI + Gradio)  
- Deploy containerizado via Docker + Docker Compose

## 🛠️ Pré‑requisitos

- Docker e Docker Compose  
- Conta OpenAI com chave configurada (`OPENAI_API_KEY`)  
- Chrome ou Firefox se for testar fora do container Selenium

## ⚙️ Instalação

```bash
git clone https://github.com/leonelmaia/virtuvia.git
cd virtuvia/app
docker compose up --build
