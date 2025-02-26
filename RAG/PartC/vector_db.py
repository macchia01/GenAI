# Librerie#

import os
import logging
import psutil
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Inizializzazione logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_memory():
    available_memory = psutil.virtual_memory().available / (1024 ** 3)
    logger.info(f"Memoria disponibile: {available_memory:.1f} GB")
    if available_memory < 4.0:
        logger.warning("Bassa memoria! Potrebbero esserci rallentamenti.")

check_memory()

# Caricamento documenti

def load_documents(docs_folder="documents"):
    all_documents = []
    
    for file in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue  # Ignora file non supportati

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # Aggiunta metadati con nome file e tipo di dato
        for chunk in chunks:
            chunk.metadata["file"] = file
            chunk.metadata["type"] = "text"

        all_documents.extend(chunks)

    logger.info(f"Caricati {len(all_documents)} chunk da {len(os.listdir(docs_folder))} file.")
    return all_documents

documents = load_documents()

# Creazione vettorstore

def create_vectorstore(documents, db_path="chromadb_index"):
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection(name="rag_collection")

    for i, doc in enumerate(documents):
        collection.add(
            documents=[doc.page_content], 
            metadatas=[doc.metadata], 
            ids=[str(i)]
        )

    logger.info("Database ChromaDB creato con successo.")
    return chroma_client, collection

chroma_client, collection = create_vectorstore(documents)
