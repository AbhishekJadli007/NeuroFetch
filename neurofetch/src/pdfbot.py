import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
import traceback
import time
import requests
import pandas as pd
import pathlib
import json

# Import all agents
from agents.structured_data_agent import StructuredDataExtractionAgent
from agents.query_reformulation_agent import QueryReformulationAgent
from agents.retrieval_agent import AdaptiveRetrievalAgent

# Initialize all agents
structured_agent = StructuredDataExtractionAgent()
query_reformulation_agent = QueryReformulationAgent()
retrieval_agent = AdaptiveRetrievalAgent()  # Will be updated with vectorstore later

def get_agent_display_name(agent_id):
    agent_names = {
        "structured_data_extraction": "📊 Structured Data Agent",
        "query_reformulation": "🔄 Query Reformulation Agent",
        "adaptive_retrieval": "🔍 Adaptive Retrieval Agent",
        "rag_system": "🤖 RAG System"
    }
    return agent_names.get(agent_id, f"Agent: {agent_id}")

def get_user_info():
    try:
        params = st.query_params
        token = params.get("token")
        if isinstance(token, list):
            token = token[0]
        if not token:
            return None
        response = requests.get(
            "http://localhost:3000/api/auth/verify",
            headers={"Authorization": f"Bearer {token}"}
        )
        if response.ok:
            user_data = response.json()
            return user_data.get("user", {})
        else:
            return None
    except Exception as e:
        return None

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.markdown(
        """
        <meta http-equiv="refresh" content="0; url='http://localhost:5173'" />
        <script>
            window.location.replace('http://localhost:5173');
        </script>
        """,
        unsafe_allow_html=True
    )
    st.experimental_rerun()

def get_document_text_and_images(uploaded_files):
    all_text = ""
    temp_dir = pathlib.Path("./temp_uploaded_files")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for uploaded_file in uploaded_files:
        temp_file_path = temp_dir / uploaded_file.name
        file_extension = temp_file_path.suffix.lower()
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"Error saving temporary file {uploaded_file.name}: {e}")
            continue
        loader = None
        try:
            if file_extension == ".pdf":
                pdf_reader = PdfReader(temp_file_path)
                pdf_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                all_text += pdf_text + "\n"
            elif file_extension == ".csv":
                from langchain_community.document_loaders import CSVLoader
                loader = CSVLoader(file_path=str(temp_file_path), encoding="utf-8")
            elif file_extension == ".txt" or file_extension == ".md":
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(file_path=str(temp_file_path), encoding="utf-8")
            else:
                st.warning(f"Unsupported file type: {file_extension}. Skipping {uploaded_file.name}.")
                continue
            if loader:
                documents = loader.load()
                loader_text = "\n".join(doc.page_content for doc in documents)
                all_text += loader_text + "\n"
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    try:
        if temp_dir.exists() and not list(temp_dir.iterdir()):
            temp_dir.rmdir()
    except OSError:
        pass
    return all_text, []

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        if not text_chunks:
            st.warning("No text chunks found to create vector store.")
            return None
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_conversation_chain(vectorstore):
    try:
        llm = OllamaLLM(model="llama3", temperature=0.5)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        if vectorstore:
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                memory=memory
            )
        return None
    except Exception as e:
        st.error(f"Error creating conversation chain: {e}")
        return None

def handle_user_input(user_question):
    if st.session_state.conversation is None and not st.session_state.get('current_pdf_filename'):
        st.error("Please process your documents first using the sidebar before asking questions.")
        st.session_state.chat_history.append({'role': 'bot', 'content': '❗ No document processed. Please upload and process a PDF first.'})
        display_chat_history()
        return
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({'role': 'user', 'content': user_question})
    with st.spinner("🔄 Reformulating query..."):
        reformulated_query = query_reformulation_agent.process({
            "query": user_question,
            "context": "document_qa"
        })
        if reformulated_query["success"]:
            final_query = reformulated_query["data"]["primary_query"]
            st.session_state.chat_history.append({
                'role': 'system', 
                'content': f"Query reformulated by {get_agent_display_name('query_reformulation')}"
            })
        else:
            final_query = user_question
    data_type = structured_agent.detect_data_type(final_query)
    pdf_filename = st.session_state.get('current_pdf_filename')
    pdf_path = os.path.join("temp_uploaded_files", pdf_filename) if pdf_filename else None
    if data_type in ["table", "chat"] and pdf_path and os.path.exists(pdf_path):
        with st.spinner(f"📊 Extracting {data_type}s using {get_agent_display_name('structured_data_extraction')}..."):
            result = structured_agent.process({
                "pdf_path": pdf_path,
                "data_type": data_type,
                "pages": "all"
            })
            agent_used = get_agent_display_name(result.get("agent_id", "structured_data_extraction"))
            if result["success"]:
                if data_type == "table" and result["data"].get("tables"):
                    for i, table in enumerate(result["data"]["tables"]):
                        df = pd.DataFrame(table["data"]) if isinstance(table, dict) and "data" in table else pd.DataFrame(table)
                        st.session_state.chat_history.append({
                            'role': 'bot',
                            'content': f'Table {i+1}:<br>{df.to_html(index=False, classes="table-auto w-full text-xs")}',
                        })
                elif data_type == "chat" and result["data"].get("chat_segments"):
                    for segment in result["data"]["chat_segments"]:
                        st.session_state.chat_history.append({
                            'role': 'bot',
                            'content': f'<pre>{segment}</pre>',
                        })
                else:
                    st.session_state.chat_history.append({
                        'role': 'bot',
                        'content': f'❗ No {data_type}s could be extracted from the PDF. (Agent: {agent_used})'
                    })
            else:
                error_msg = result.get('error', f'Failed to extract {data_type}s.')
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': f'❗ Extraction error: {error_msg}'
                })
        display_chat_history()
        return
    if st.session_state.conversation:
        with st.spinner(f"🔍 Searching documents using {get_agent_display_name('adaptive_retrieval')}..."):
            retrieval_result = retrieval_agent.process({
                "queries": [final_query],
                "original_query": final_query
            })
            if retrieval_result["success"]:
                enhanced_query = final_query
                st.session_state.chat_history.append({
                    'role': 'system', 
                    'content': f"Query processed by {get_agent_display_name('adaptive_retrieval')}"
                })
            else:
                enhanced_query = final_query
        try:
            with st.spinner("🤖 Generating response..."):
                start_time = time.time()
                response = st.session_state.conversation({'question': enhanced_query})
                end_time = time.time()
                fetch_time = end_time - start_time
                response_content = response['answer']
                agent_used = get_agent_display_name('rag_system')
                response_with_agent = f"{response_content}\n\n---\n*Response generated by {agent_used} in {fetch_time:.2f} seconds*"
                st.session_state.chat_history.append({
                    'role': 'bot', 
                    'content': response_with_agent
                })
        except Exception as e:
            st.session_state.chat_history.append({
                'role': 'bot',
                'content': f'❗ Error processing your question: {str(e)}'
            })
            st.error(f"Error processing your question: {str(e)}")
            import sys
            print("--- Detailed Error Traceback ---")
            traceback.print_exc(file=sys.stdout)
            print("------------------------------")
            if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
                st.session_state.chat_history.pop()
    else:
        st.session_state.chat_history.append({'role': 'bot', 'content': '❗ No conversation chain available. Please process documents first.'})
        st.error("No conversation chain available. Please process documents first.")
    display_chat_history()

def display_chat_history():
    chat_messages_html = ""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                chat_messages_html += f'''
                <div class="flex items-start space-x-4 mb-8">
                    <img alt="User avatar" class="w-8 h-8 rounded-full" src="https://i.ibb.co/C8y3Gz2/user-avatar.png"/>
                    <div>
                        <p class="font-semibold text-gray-800 dark:text-gray-200">You</p>
                        <div class="bg-white p-4 rounded-lg mt-1 shadow dark:bg-gray-700">
                            <p class="text-sm text-gray-800 dark:text-gray-200">{message['content']}</p>
                        </div>
                    </div>
                </div>
                '''
            elif message['role'] == 'system':
                chat_messages_html += f'''
                <div class="flex items-start space-x-4 mb-4">
                    <img alt="System avatar" class="w-8 h-8 rounded-full bg-blue-500 p-1" src="https://i.ibb.co/hK8b7XW/system-avatar.png"/>
                    <div>
                        <p class="font-semibold text-blue-500">System</p>
                        <div class="bg-blue-100 p-2 rounded-lg mt-1 shadow dark:bg-blue-900/50">
                            <p class="text-xs text-blue-800 dark:text-blue-200">{message['content']}</p>
                        </div>
                    </div>
                </div>
                '''
            else:  # role == 'bot'
                message_key = f"bot_message_{len(st.session_state.chat_history)}"
                chat_messages_html += f'''
                <div class="flex items-start space-x-4 mb-8">
                    <img alt="AI avatar" class="w-8 h-8 rounded-full bg-indigo-500 p-1" src="https://i.ibb.co/kH65yqF/ai-avatar.png"/>
                    <div>
                        <p class="font-semibold text-gray-800 dark:text-gray-200">CHAT <span class="text-indigo-500">A.I+</span></p>
                        <div class="bg-white p-4 rounded-lg mt-1 shadow dark:bg-gray-700">
                            <div id="{message_key}"></div> </div>
                        <div class="flex items-center space-x-4 mt-3">
                            <button class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                                <span class="material-icons">thumb_up</span>
                            </button>
                            <button class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                                <span class="material-icons">thumb_down</span>
                            </button>
                            <button class="flex items-center text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                                <span class="material-icons mr-1 text-base">refresh</span>
                                Regenerate
                            </button>
                        </div>
                    </div>
                </div>
                '''
    st.session_state.chat_display_html = chat_messages_html

def main():
    st.set_page_config(page_title="NeuroFetch PDFBot", page_icon="🤖", layout="wide")
    # Inject custom CSS
    with open(os.path.join(os.path.dirname(__file__), "streamlit_custom.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "images" not in st.session_state:
        st.session_state.images = []
    if "current_pdf_filename" not in st.session_state:
        st.session_state.current_pdf_filename = None
    if "user_input_value" not in st.session_state:
        st.session_state.user_input_value = ""

    st.title("NeuroFetch PDFBot")
    st.write("Upload your documents and chat with them using AI-powered retrieval and extraction.")

    uploaded_files = st.file_uploader(
        "Upload (PDF, CSV, TXT, MD) and click 'Process'",
        accept_multiple_files=True,
        type=["pdf", "csv", "txt", "md"]
    )
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                raw_text, images = get_document_text_and_images(uploaded_files)
            if not raw_text.strip():
                st.error("No text could be extracted. Please check your documents.")
                st.session_state.conversation = None
                st.session_state.images = []
                st.session_state.chat_history = []
            else:
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore:
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.images = []
                    retrieval_agent.update_vectorstore(vectorstore)
                    st.success("Documents processed successfully! You can now ask questions.")
                    st.session_state.chat_history = []
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name.lower().endswith('.pdf'):
                            st.session_state['current_pdf_filename'] = uploaded_file.name
                            break
                else:
                    st.error("Failed to create conversation chain. Check Ollama setup and ensure 'llama3' model is pulled.")
                    st.session_state.conversation = None
                    st.session_state.images = []
                    st.session_state.chat_history = []
        else:
            st.warning("Please upload at least one document file.")
            st.session_state.conversation = None
            st.session_state.images = []
            st.session_state.chat_history = []

    # Chat area
    st.subheader("Chat")
    user_input = st.text_input("Type your question and press Enter", value=st.session_state.user_input_value, key="user_input")
    if st.button("Send"):
        if user_input.strip():
            handle_user_input(user_input)
            st.session_state.user_input_value = ""

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'bot':
            st.markdown(
                f'<div class="chat-message-container">'
                f'<div class="chat-avatar" style="font-size:2rem;">🤖</div>'
                f'<div class="chat-message-bot">{message["content"]}</div></div>',
                unsafe_allow_html=True
            )
        elif message['role'] == 'system':
            st.markdown(
                f'<div class="chat-message-container">'
                f'<div class="chat-avatar" style="font-size:2rem;">🤖</div>'
                f'<div class="chat-message-bot">{message["content"]}</div></div>',
                unsafe_allow_html=True
            )
        elif message['role'] == 'user':
            st.markdown(
                f'<div class="chat-message-container user">'
                f'<div class="chat-message-user">{message["content"]}</div>'
                f'<div class="chat-avatar" style="font-size:2rem;">🧑</div></div>',
                unsafe_allow_html=True
            )

if __name__ == '__main__':
    main()