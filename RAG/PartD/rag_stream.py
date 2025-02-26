################################################################
#                        RAG APP - OLLAMA                      #
################################################################
# Esecuzione:
#   -> streamlit run app.py
# Assicurati di essere nella directory corrente per il percorso della chroma db

import requests
import chromadb
import ollama
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

######################### Tool: SerperSearchTool #############################
class SerperSearchTool:
    def __init__(self, api_key, max_results=5):
        self.api_key = api_key
        self.max_results = max_results

    def search(self, query):
        headers = {"X-API-Key": self.api_key}
        payload = {"q": query, "gl": "us", "hl": "en"}
        response = requests.post("https://api.serper.dev/search", json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if "organic" in data:
                results = data["organic"][:self.max_results]
                formatted_results = "\n".join(
                    [f"{res.get('title', 'Nessun Titolo')}: {res.get('snippet', '')}" for res in results]
                )
                return formatted_results
            else:
                return "Nessun risultato trovato per la ricerca."
        else:
            return f"Errore nella ricerca: {response.status_code} {response.text}"

######################### Funzioni Aggiuntive #############################
def decide_strategy(query, model="llama3.2"):
    """
    Data una domanda, decide quale strategia adottare:
    a) Usa i documenti indicizzati.
    b) Esegui una ricerca su internet (Serper).
    c) Rispondi direttamente senza contesto esterno.
    
    Risponde solo con "a", "b" o "c".
    """
    prompt = f"""
Sei un assistente intelligente. Data la domanda:
"{query}"
decidi quale strategia adottare per rispondere:
a) Usa i documenti indicizzati per rispondere.
b) Esegui una ricerca su internet per ottenere informazioni aggiornate.
c) Rispondi direttamente senza contesto esterno.
Rispondi solo con "a", "b" o "c".
    """
    messages = [{"role": "system", "content": prompt}]
    response = list(ollama.chat(model=model, messages=messages, stream=False))
    
    # Gestione della struttura della risposta
    last_item = response[-1]
    if isinstance(last_item, tuple):
        # Se la risposta è una tupla, assumiamo che il messaggio si trovi al secondo elemento
        decision = last_item[1]["content"].strip().lower()
    else:
        decision = last_item["message"]["content"].strip().lower()
    
    return decision


def respond_with_context(lst_messages, additional_context, model="llama3.2"):
    """
    Risponde integrando un contesto aggiuntivo (es. risultati Serper).
    """
    q = lst_messages[-1]["content"]
    prompt = f"""
Sei un assistente per compiti di domande e risposte.
Utilizza il seguente contesto aggiuntivo per rispondere alla domanda:

#### Contesto Aggiuntivo (Risultati Serper) ####
{additional_context}

#### Domanda ####
{q}

Fornisci una risposta concisa basata su questo contesto.
    """
    messages = [{"role": "system", "content": prompt}] + lst_messages
    res_ai = ollama.chat(model=model, messages=messages, stream=True)
    for res in res_ai:
        chunk = res["message"]["content"]
        app["full_response"] += chunk
        yield chunk

def direct_respond(lst_messages, model="llama3.2"):
    """
    Risponde direttamente senza alcun contesto esterno.
    """
    q = lst_messages[-1]["content"]
    prompt = f"""
Sei un assistente che risponde direttamente alle domande.
Rispondi alla seguente domanda in maniera completa e chiara:

{q}
    """
    messages = [{"role": "system", "content": prompt}] + lst_messages
    res_ai = ollama.chat(model=model, messages=messages, stream=True)
    for res in res_ai:
        chunk = res["message"]["content"]
        app["full_response"] += chunk
        yield chunk

######################## Backend ##############################
class AI():
    def __init__(self):
        self.db = chromadb.PersistentClient(path="chromadb_index")
        self.collection = self.db.get_or_create_collection("rag_collection")

    def query(self, q, top=10):
        res_db = self.collection.query(query_texts=[q])["documents"][0][0:top]
        context = ' '.join(res_db).replace("\n", " ")
        return context

    def respond(self, lst_messages, model="llama3.2"):
        """
        Risponde basandosi sul contesto recuperato dai documenti indicizzati.
        """
        q = lst_messages[-1]["content"]
        context = self.query(q)
        prompt = f"""
Sei un assistente per domande e risposte.
Utilizza il seguente contesto recuperato dai documenti per rispondere alla domanda.
Se il contesto non contiene le informazioni necessarie, non inventare risposte.

#### Contesto Recuperato ####
{context}

#### Domanda ####
{q}

Fornisci una risposta concisa basata su questo contesto.
        """
        messages = [{"role": "system", "content": prompt}] + lst_messages
        res_ai = ollama.chat(model=model, messages=messages, stream=True)
        for res in res_ai:
            chunk = res["message"]["content"]
            app["full_response"] += chunk
            yield chunk

ai = AI()

# Inserisci qui la tua chiave API Serper
SERPER_API_KEY = "03564002e814cbcbe126d6dfaa1a7a4403047b3c"

serper_tool = SerperSearchTool(api_key=SERPER_API_KEY, max_results=5)

######################## Frontend #############################
st.title('💬 Chiedi le tue domande')
st.sidebar.title("Caricamento File & Cronologia Chat")
app = st.session_state

# Gestione caricamento file
uploaded_files = st.sidebar.file_uploader("Carica i tuoi file (PDF o TXT)", accept_multiple_files=True)
if uploaded_files:
    all_documents = []
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(uploaded_file)
            documents = loader.load()
        elif uploaded_file.name.endswith(".txt"):
            documents = [Document(page_content=file_content, metadata={"file": uploaded_file.name, "type": "text"})]
        else:
            continue
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        all_documents.extend(chunks)
    chroma_client = chromadb.PersistentClient(path="chromadb_index")
    collection = chroma_client.get_or_create_collection(name="rag_collection")
    for i, doc in enumerate(all_documents):
        collection.add(
            documents=[doc.page_content],
            metadatas=[doc.metadata],
            ids=[str(i)]
        )
    st.sidebar.success("Database aggiornato con successo!")

if "messages" not in app:
    app["messages"] = [{"role": "assistant", "content": "Risponderò basandomi esclusivamente sul contesto recuperato. Se non trovo contesto, utilizzerò altre strategie."}]
if "history" not in app:
    app["history"] = []
if "full_response" not in app:
    app["full_response"] = ''

for msg in app["messages"]:
    avatar = "😎" if msg["role"] == "user" else "👾"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if txt := st.chat_input():
    app["messages"].append({"role": "user", "content": txt})
    st.chat_message("user", avatar="😎").write(txt)
    app["full_response"] = ""
    strategy = decide_strategy(txt)
    st.sidebar.markdown(f"**Strategia scelta:** {strategy.upper()}")
    if strategy == "a":
        ai_response = ai.respond(app["messages"])
    elif strategy == "b":
        search_results = serper_tool.search(txt)
        ai_response = respond_with_context(app["messages"], additional_context=search_results)
    elif strategy == "c":
        ai_response = direct_respond(app["messages"])
    else:
        ai_response = direct_respond(app["messages"])
    st.chat_message("assistant", avatar="👾").write_stream(ai_response)
    app["messages"].append({"role": "assistant", "content": app["full_response"]})
    app["history"].append(f"😎: {txt}")
    app["history"].append(f"👾: {app['full_response']}")
    st.sidebar.markdown("<br />".join(app["history"]) + "<br /><br />", unsafe_allow_html=True)
