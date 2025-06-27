from utils import OpenAI
import logging

logging.basicConfig(level=logging.INFO)


def classify_case_gpt(text: str, openai: OpenAI) -> str:
    prompt = f"""
Classifique o seguinte texto jurídico em uma das categorias abaixo:
- Trabalhista
- Cível
- Tributária
- Penal
- Previdenciária
- Outros

Texto:
\"\"\"
{text}
\"\"\"

Responda apenas com a categoria.
"""
    response = openai.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response.choices[0].message.content.strip()


def extract_keyword(msg: str, openai: OpenAI) -> str:
    prompt = f"""
    Extraia a principal palavra ou expressão-chave jurídica deste texto para ser usada em uma busca legal:
    
    Texto:
    \"\"\"
    "{msg}"
    \"\"\"
    Responda apenas com a palavra ou expressão, sem explicações.
    """
    response = openai.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return response.choices[0].message.content.strip()


def analyze_risk(case_type: str, amount: float) -> dict:
    base_probs = {
        "Trabalhista": 0.65,
        "Cível": 0.45,
        "Tributária": 0.60,
        "Penal": 0.30,
        "Previdenciária": 0.70,
        "Others": 0.50,
    }
    case_type_normalized = case_type.strip().lower()
    chance = base_probs.get(case_type, 0.5)
    risk_level = "Baixo" if chance > 0.6 else "Medio" if chance > 0.4 else "Alto"
    estimated_loss = round(amount * (1 - chance), 2)
    return {
        "Tipo": case_type_normalized,
        "Probabilidade de sucesso": round(chance * 100, 2),
        "Risco": risk_level,
        "estimativa_perda": estimated_loss,
    }


def generate_mock_report(area: str) -> dict:

    mock_data = {
        "trabalhista": {
            "casos": 52,
            "vitorias": 35,
            "acordos": 10,
            "tempo_medio": "91 days",
        },
        "cível": {"casos": 29, "vitorias": 15, "acordos": 7, "tempo_medio": "88 days"},
        "penal": {"casos": 8, "vitorias": 3, "acordos": 2, "tempo_medio": "122 days"},
    }

    normalized_area = area.strip().lower()
    return mock_data.get(normalized_area, {"message": "Area not found."})
