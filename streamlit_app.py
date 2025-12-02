import os
import json
#from dotenv import load_dotenv
import streamlit as st
from embedchain import App

# ==========================
# Config e helper Embedchain
# ==========================

#load_dotenv()  # per usare OPENAI_API_KEY da .env

openai_key = os.getenv("OPENAI_API_KEY", "NON TROVATA")
if openai_key != 'NON TROVATA':
    st.write(f"OPENAI_API_KEY presente? {'sì' if openai_key != 'NON TROVATA' else 'no'}")

EMBEDCHAIN_CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-mini",   # o "gpt-4o-mini" a seconda di cosa usi
            "temperature": 0.2,
            "max_tokens": 900,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
        },
    },
    # niente vectordb configurato -> usa quello di default (in-memory)
}


def crea_app_embedchain() -> App:
    """Crea una nuova istanza di Embedchain App con la nostra config."""
    return App.from_config(config=EMBEDCHAIN_CONFIG)


def genera_presentazione_da_sito(url: str, output_format: str = "text") -> str:
    """
    Genera una presentazione aziendale dal sito indicato.
    output_format: "text" oppure "json"
    Ritorna sempre una stringa (testo o JSON raw).
    """
    if not url.startswith("http"):
        url = "https://" + url

    app = crea_app_embedchain()

    # Aggiungo il sito come sorgente
    app.add(url, data_type="web_page")

    # Prompt "base" condiviso
    prompt_base = f"""
Prendi SOLO le informazioni presenti nel sito: {url}
NON inventare dati: se qualcosa non è chiaro, scrivi "Non specificato".
"""

    # Prompt per JSON
    prompt_json = prompt_base + """
Genera la presentazione aziendale in formato JSON con ESATTAMENTE questa struttura:

{
  "nome_azienda": "",
  "settore": "",
  "descrizione": "",
  "servizi_prodotti": [],
  "clienti_riferimento": [],
  "punti_di_forza": [],
  "area_geografica": ""
}

Non aggiungere testo fuori dal JSON, niente commenti, niente markdown.
"""

    # Prompt per testo
    prompt_testo = prompt_base + """
Genera una presentazione aziendale strutturata in testo con questa forma:

**Nome azienda**
[Nome azienda]

**Settore**
[Settore]

**Cosa fa l'azienda**
[Breve descrizione in 3-5 frasi]

**Servizi / Prodotti principali**
- punto 1
- punto 2
- punto 3

**Clienti di riferimento**
- tipologia 1
- tipologia 2

**Punti di forza**
- punto 1
- punto 2
- punto 3

**Area geografica di riferimento**
[indicazione]

Usa frasi brevi e tono professionale.
"""

    if output_format == "json":
        risposta = app.query(prompt_json)
    else:
        risposta = app.query(prompt_testo)

    return risposta


# ==========================
# Frontend Streamlit
# ==========================

def main():
    st.set_page_config(page_title="GEMMA - Presentazione aziendale da sito web", page_icon="🏢", layout="centered")

    st.title("🏢 GEMMA - Generatore di presentazione aziendale da sito web")
    st.write("Inserisci l'URL del sito aziendale e scegli il formato di output.")

    url = st.text_input("URL del sito", placeholder="https://www.esempio.it")

    formato = st.radio(
        "Formato output",
        options=["Testo", "JSON"],
        index=0,
        horizontal=True,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        genera = st.button("Genera presentazione")

    if genera:
        if not url:
            st.warning("Per favore inserisci un URL.")
            return

        output_format = "json" if formato == "JSON" else "text"

        with st.spinner("Sto analizzando il sito e generando la presentazione..."):
            try:
                raw_output = genera_presentazione_da_sito(url, output_format=output_format)
            except Exception as e:
                st.error(f"Errore durante la generazione: {e}")
                return

        st.subheader("Risultato")

        if output_format == "text":
            # Mostro come markdown, pronto da copiare
            st.markdown(raw_output)
        else:
            # Provo a fare parse del JSON per mostrarlo bene
            try:
                json_obj = json.loads(raw_output)
                st.json(json_obj)

                with st.expander("Vedi JSON raw"):
                    st.code(raw_output, language="json")
            except json.JSONDecodeError:
                st.warning("Il modello non ha restituito un JSON perfettamente valido. Ecco l'output raw:")
                st.code(raw_output, language="json")


if __name__ == "__main__":
    main()
