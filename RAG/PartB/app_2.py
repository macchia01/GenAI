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

######################## Backend ##############################
class AI():
    def __init__(self):
        self.db = chromadb.PersistentClient(path="RAG\PartB\chromadb_index") 
        self.collection = self.db.get_or_create_collection("rag_collection")

    def query(self, q, top=10):
        res_db = self.collection.query(query_texts=[q])["documents"][0][0:top]
        context = ' '.join(res_db).replace("\n", " ")
        return context

    def respond(self, lst_messages, model="llama3.2"):  # Removed use_knowledge flag
        q = lst_messages[-1]["content"]
        context = self.query(q)

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

######################## Frontend #############################
st.title('💬 Ask your questions')
st.sidebar.title("Chat History")
app = st.session_state

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
    app["messages"].append({"role": "user", "content": txt})
    st.chat_message("user", avatar="😎").write(txt)
    
    ## AI response
    app["full_response"] = ""
    st.chat_message("assistant", avatar="👾").write_stream(ai.respond(app["messages"]))
    app["messages"].append({"role": "assistant", "content": app["full_response"]})
    
    ## Show history
    app['history'].append(f"😎: {txt}")
    app['history'].append(f"👾: {app['full_response']}")
    st.sidebar.markdown("<br />".join(app['history']) + "<br /><br />", unsafe_allow_html=True)
