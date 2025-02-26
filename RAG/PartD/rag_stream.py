################################################################
#                        RAG APP - OLLAMA                      #
################################################################

## -> streamlit run app.py
## -> check the current working directory to set the path for the chroma db

## for db
import chromadb
## for ai
import ollama
## for app
import streamlit as st
## for document processing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

######################## Backend ##############################
class AI():
    def __init__(self):
        self.db = chromadb.PersistentClient(path="chromadb_index") 
        self.collection = self.db.get_or_create_collection("rag_collection")

    def query_documents(self, q, top=10):
        if self.collection.count() == 0:
            return "No documents found in the database."
        
        res_db = self.collection.query(query_texts=[q])["documents"][0][0:top]
        context = ' '.join(res_db).replace("\n", " ")
        return context

    def respond(self, lst_messages, model="llama3.2"):
        q = lst_messages[-1]["content"]
        context = self.query_documents(q)
        if context == "No documents found in the database.":
            context = ""

        q = lst_messages[-1]["content"]
        context = self.query_documents(q)

        template = """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question:
        If you don't know the answer from the context, then do not answer from your own knowledge.
        Keep the answer concise.

        #### Retrieved Context ####
        {context}

        #### Question ####
        {question}

        #### LLM Response ####
        """
        prompt = template.format(context=context, question=q)
        
        res_ai = ollama.chat(model=model, messages=[{"role": "system", "content": prompt}] + lst_messages, stream=True)
        
        for res in res_ai:
            chunk = res["message"]["content"]
            app["full_response"] += chunk
            yield chunk

ai = AI()

# Define tools for the agent
search_tool = Tool(
    name="Web Search",
    func=DuckDuckGoSearchAPIWrapper().run,
    description="Use this tool when the user asks for the latest information from the web."
)

document_search_tool = Tool(
    name="Document Search",
    func=ai.query_documents,
    description="Use this tool to search among uploaded documents for relevant information."
)

assistant_tool = Tool(
    name="AI Assistant",
    func=lambda query: '
'.join(ai.respond([{"role": "user", "content": query}])), 
    description="Use this tool for general conversations and assistance."
)

agent = initialize_agent(
    tools=[search_tool, document_search_tool, assistant_tool],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

######################## Frontend #############################
st.title('💬 Ask your questions')
st.sidebar.title("Upload & Chat History")
app = st.session_state

# Gestione caricamento file
uploaded_files = st.sidebar.file_uploader("Carica i tuoi file (PDF o TXT)", accept_multiple_files=True)

if uploaded_files:
    all_documents = []
    
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.read().decode("utf-8")  # Read content from uploaded file
        
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(uploaded_file)
            documents = loader.load()
        elif uploaded_file.name.endswith(".txt"):
            documents = [Document(page_content=file_content, metadata={"file": uploaded_file.name, "type": "text"})]
        else:
            continue  # Ignora file non supportati
        
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
    app["messages"] = [{"role": "assistant", "content": "I will only answer based on retrieved context. If no context is found, I will not respond."}]

if 'history' not in app:
    app['history'] = []

if 'full_response' not in app:
    app['full_response'] = ''

## Display previous messages
for msg in app["messages"]:
    st.chat_message(msg["role"], avatar=("😎" if msg["role"] == "user" else "👾")).write(msg["content"])

## Chat input
if txt := st.chat_input():
    response = agent.run(txt)
    app["messages"].append({"role": "user", "content": txt})
    st.chat_message("user", avatar="😎").write(txt)
    
    ## AI response
    app["full_response"] = response
    st.chat_message("assistant", avatar="👾").write(response)
    app["messages"].append({"role": "assistant", "content": response})
    
    ## Show history
    app['history'].append(f"😎: {txt}")
    app['history'].append(f"👾: {response}")
    st.sidebar.markdown("<br />".join(app['history']) + "<br /><br />", unsafe_allow_html=True)
