from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

def Q_A(vectorstore,question):
    ollama_llm = ChatOllama(
    model="llama3.1",
    temperature=0.5,
)
    qa = RetrievalQA.from_chain_type(llm=ollama_llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    answer = qa.invoke(question)

    return answer['result']