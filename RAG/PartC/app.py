import streamlit as st
import ollama
from utils.QA_Bot import QA_Bot
from utils.PDF_Reader import PDF_4_QA


# Assicurati di usare llama3.2 (già installato localmente)
ollama.pull("llama3.2")

# Streamlit app
def main():
    st.sidebar.title("Carica un file PDF")
    
    uploaded_file = st.sidebar.file_uploader("Scegli un file PDF", type="pdf")
    
    if uploaded_file is not None:
        st.sidebar.success("File caricato con successo.")
        vector_store = PDF_4_QA(uploaded_file)
        QA_Bot(vector_store)
    else:
        st.sidebar.warning("Carica un file PDF per iniziare.")

if __name__ == '__main__':
    main()
