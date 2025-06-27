import os
from glob import glob
import gradio as gr
from utils import (
    OpenAI,
    QdrantClient,
    get_context_from_qdrant,
    build_messages,
    filter_relevant_sources,
    add_pdf_to_qdrant_index,
    rebuild_doc_map,
    delete_pdf_from_qdrant,
    move_pdf_files,
    remove_temp_gradio_file,
    process_documents_lexml,
)
import markdown
import pandas as pd
from functools import partial
import logging
import requests
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO)

load_dotenv() 
openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
qdrant = QdrantClient(host="qdrant", port=6333)
DOCS_PATH = "documents"
API_URL = "http://localhost:8000"

doc_map = rebuild_doc_map(qdrant, collection_name="docs")


def get_pdf_list():
    return [os.path.basename(f) for f in glob("documents/*.pdf")]


def delete_pdf_gradio(selected_pdf, qdrant, doc_map=doc_map):
    if not selected_pdf:
        return "⚠️ Nenhum PDF selecionado.", gr.update()

    status = delete_pdf_from_qdrant(selected_pdf, qdrant, doc_map=doc_map)
    status = remove_temp_gradio_file(selected_pdf)
    updated_list = get_pdf_list()

    return (status, gr.update(choices=updated_list, value=None))


def upload_pdf(file, openai=openai, qdrant=qdrant, doc_map=doc_map):
    file_name = os.path.basename(file)
    if file is None:
        return "Nenhum arquivo selecionado."

    try:
        move_pdf_files(file)
        add_pdf_to_qdrant_index(
            pdf_path=file_name,
            doc_map=doc_map,
            openai=openai,
            qdrant=qdrant,
            chunk_size=5096,
            overlap=512,
            collection_name="docs",
        )
        return f"✅ PDF '{os.path.basename(file.name)}' importado com sucesso para o Qdrant."
    except Exception as e:
        logging.info(f"❌ Erro ao importar o PDF: {str(e)}")
        return


def render_chat(history, sources_text=""):
    html = ""
    for user_msg, bot_msg_md in history:
        user_html = f'<div style="text-align:right; margin:10px; padding:8px; background:#d0f0fd; border-radius:8px; color: black !important;">{markdown.markdown(user_msg)}</div>'
        bot_html = f'<div style="text-align:left; margin:10px; padding:8px; background:#f0f0f0; border-radius:8px; color: black !important;">{markdown.markdown(bot_msg_md)}</div>'
        html += user_html + bot_html

    if sources_text:
        sources_html = f"""
        <div style="margin:10px; padding:8px; background:#fff9c4; border-radius:8px; color: black !important;">
            <b>Fontes usadas:</b><br>{sources_text}
        </div>
        """
        html += sources_html

    return html


def chat_with_context(user_input, history):
    history = history or []

    if user_input.lower().startswith("/federal"):
        text = user_input[len("/federal") :].strip()
        try:
            response = requests.post(f"{API_URL}/law", json={"text": text})
            response.raise_for_status()
            keyword = response.json()["keyword"]
            reply = f"📂 Segundo a jurisprudencia federal: **{process_documents_lexml(keyword,openai,qdrant)}**"
        except Exception as e:
            reply = f"❌ Error: {str(e)}"

    elif user_input.lower().startswith("/classificar"):
        text = user_input[len("classificar") :].strip()
        try:
            response = requests.post(f"{API_URL}/classify", json={"text": text})
            response.raise_for_status()
            category = response.json()["category"]
            reply = f"📂 Case classified as: **{category}**"
        except Exception as e:
            reply = f"❌ Error: {str(e)}"
        history.append((user_input, reply))
        return "", history, render_chat(history)

    elif user_input.lower().startswith("/risco"):
        try:
            _, case_type, value = user_input.split(":", 2)
            response = requests.post(
                f"{API_URL}/risk",
                json={"case_type": case_type.strip(), "value": float(value)},
            )
            response.raise_for_status()
            result = response.json()
            logging.info(f"Creating risk=> {result}")
            reply = f"""📊 **Risk Analysis**
- Type: {result['Tipo']}
- Success Probability: {result['Probabilidade de sucesso']}%
- Risk Level: {result['Risco']}
- Estimated Loss: R$ {result['estimativa_perda']:.2f}
"""
        except Exception as e:
            reply = f"❌ Error: {str(e)}"
        history.append((user_input, reply))
        return "", history, render_chat(history)

    elif user_input.lower().startswith("/relatorio"):
        area = user_input[len("/relatorio ") :].strip()
        try:
            response = requests.post(f"{API_URL}/report", json={"area": area})
            response.raise_for_status()
            result = response.json()
            if "message" in result:
                reply = f"⚠️ {result['message']}"
            else:
                reply = f"""📈 **Report for {area.title()} Area**
- Total Cases: {result['casos']}
- Wins: {result['vitorias']}
- Settlements: {result['acordos']}
- Average Duration: {result['tempo_medio']}
"""
        except Exception as e:
            reply = f"❌ Error: {str(e)}"
        history.append((user_input, reply))
        return "", history, render_chat(history)

    structured_history = [{"user": u, "assistant": a} for u, a in history]

    results = get_context_from_qdrant(user_input, openai=openai, qdrant=qdrant, k=3)
    logging.info(f"Creating results=> {results}")

    context = "\n\n".join([chunk for _, _, chunk in results])

    messages = build_messages(context, structured_history, user_input)

    response = openai.chat.completions.create(model="gpt-4o", messages=messages)
    assistant_reply = response.choices[0].message.content
    unique_sources = set()
    relevant_sources = filter_relevant_sources(
        assistant_reply, results, openai, threshold=0.80
    )

    if len(relevant_sources) > 0:
        unique_sources = set(relevant_sources)
        sources_text = "<br>".join(
            f"[Fonte: {s}, Página {p}]" for s, p in sorted(unique_sources)
        )
    else:
        sources_text = ""

    history.append((user_input, assistant_reply))
    chat_html = render_chat(history, sources_text)

    return "", history, chat_html


def update_dropdown():
    return gr.update(choices=get_pdf_list())


with gr.Blocks(
    css="""
    html, body, .gradio-container {
        height: 100vh !important;
        margin: 0;
        padding: 0;
        overflow-y: auto;
        display: flex;
    }
    #chat_wrapper {
        width: 80vw;
        height: 60vh;
        display: flex;
        flex-direction: column;
        border: 1px solid #ccc;
        border-radius: 1px;
        background: transparent;
        box-shadow: 0 1px 1px rgb(0 0 0 / 0.1);
    }
    #chat_container {
        flex-grow: 1;
        overflow-y: auto;
        padding: 10px;
    }
    #input_container {
        height: 65px;
        border-top: 1px solid #ccc;
        display: flex;
        box-sizing: border-box;
    }
    #input_container textarea {
        flex-grow: 1;
        resize: none;
    }
    #upload_column, #delete_column {
        max-height: 600px;   /* ajuste conforme preferir */
        overflow-y: auto;
        padding-right: 10px;
        border: 1px solid #ddd;
        border-radius: 8px;
}
"""
) as demo:
    with gr.Tabs():
        with gr.TabItem("Chat"):

            gr.HTML(
                """
            <style>
                #chat_container * {
                    color: black !important;
                }
            </style>
            """
            )
            gr.Image(
                value="images/da-vinci.png",
                width=50,
                height=50,
                show_label=False,
                container=False,
            )
            gr.Markdown("### Welcome to VirtuvIA - AI Document Chatbot")

            with gr.Column(elem_id="chat_wrapper"):
                chat_html = gr.HTML(value="", elem_id="chat_container")
                with gr.Row(elem_id="input_container"):
                    user_input = gr.Textbox(
                        placeholder="Type your message...", show_label=False
                    )

            state = gr.State([])

            user_input.submit(
                chat_with_context,
                inputs=[user_input, state],
                outputs=[user_input, state, chat_html],
            )

        with gr.Tab("📄 Gerenciar arquivos"):
            with gr.Row():
                with gr.Column(elem_id="upload_column", scale=1):
                    gr.Markdown("### ➕ Enviar novo PDF para o Assistente")
                    file_upload = gr.File(label="Escolher PDF", file_types=[".pdf"])
                    upload_btn = gr.Button("Enviar para o Assistente")
                    upload_output = gr.Textbox(label="Status do Upload")
                with gr.Column(elem_id="delete_column", scale=1):
                    gr.Markdown("### 🗑️ Remover PDF do Assistente")
                    delete_dropdown = gr.Dropdown(
                        choices=[], label="Escolha o PDF para remover"
                    )
                    delete_btn = gr.Button("Remover do Assistente")
                    delete_output = gr.Textbox(label="Status da Remoção")

            refresh_timer = gr.Timer(1.0)

        refresh_timer.tick(fn=update_dropdown, outputs=delete_dropdown)
        upload_btn.click(
            fn=partial(upload_pdf, openai=openai, qdrant=qdrant, doc_map=doc_map),
            inputs=[file_upload],
            outputs=upload_output,
        )
        delete_btn.click(
            fn=partial(delete_pdf_gradio, qdrant=qdrant),
            inputs=[delete_dropdown],
            outputs=[delete_output, delete_dropdown],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
