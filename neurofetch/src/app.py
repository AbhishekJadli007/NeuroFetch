from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
import pathlib
import traceback
import time
import json
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import requests
import logging

# Import all agents
from agents.structured_data_agent import StructuredDataExtractionAgent
from agents.query_reformulation_agent import QueryReformulationAgent
from agents.retrieval_agent import AdaptiveRetrievalAgent

app = Flask(__name__)
CORS(app)

# Global variables to store conversation state
conversation_chain = None
vectorstore = None
current_pdf_filename = None
chat_history = []

# Initialize all agents
structured_agent = StructuredDataExtractionAgent()
query_reformulation_agent = QueryReformulationAgent()
retrieval_agent = AdaptiveRetrievalAgent()

MCP_SERVER_URL = "http://localhost:8000/route_query"
MCP_AGENTS_URL = "http://localhost:8000/agents"
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FlaskBackend")

def get_agent_display_name(agent_id):
    agent_names = {
        "structured_data_extraction": "📊 Structured Data Agent",
        "query_reformulation": "🔄 Query Reformulation Agent",
        "adaptive_retrieval": "🔍 Adaptive Retrieval Agent",
        "rag_system": "🤖 RAG System"
    }
    return agent_names.get(agent_id, f"Agent: {agent_id}")

def get_document_text_and_images(uploaded_files):
    all_text = ""
    temp_dir = pathlib.Path("./temp_uploaded_files")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for uploaded_file in uploaded_files:
        temp_file_path = temp_dir / uploaded_file.filename
        file_extension = temp_file_path.suffix.lower()
        
        try:
            uploaded_file.save(temp_file_path)
        except Exception as e:
            print(f"Error saving temporary file {uploaded_file.filename}: {e}")
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
                print(f"Unsupported file type: {file_extension}. Skipping {uploaded_file.filename}.")
                continue
                
            if loader:
                documents = loader.load()
                loader_text = "\n".join(doc.page_content for doc in documents)
                all_text += loader_text + "\n"
        except Exception as e:
            print(f"Error processing file {uploaded_file.filename}: {e}")
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
            print("No text chunks found to create vector store.")
            return None
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        print(f"Error creating vector store: {e}")
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
        print(f"Error creating conversation chain: {e}")
        return None

@app.route('/api/upload', methods=['POST'])
def upload_files():
    global conversation_chain, vectorstore, current_pdf_filename, chat_history
    
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
            
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'success': False, 'error': 'No files selected'}), 400
            
        # Process documents
        raw_text, images = get_document_text_and_images(files)
        
        if not raw_text.strip():
            return jsonify({'success': False, 'error': 'No text could be extracted from the documents'}), 400
            
        # Create vector store
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        
        if not vectorstore:
            return jsonify({'success': False, 'error': 'Failed to create vector store'}), 500
            
        # Create conversation chain
        conversation_chain = get_conversation_chain(vectorstore)
        
        if not conversation_chain:
            return jsonify({'success': False, 'error': 'Failed to create conversation chain'}), 500
            
        # Update retrieval agent
        retrieval_agent.update_vectorstore(vectorstore)
        
        # Reset chat history
        chat_history = []
        
        # Set current PDF filename if any PDF was uploaded
        for file in files:
            if file.filename.lower().endswith('.pdf'):
                current_pdf_filename = file.filename
                break
        else:
            current_pdf_filename = None
            
        # Here, update your vectorstore with the new files
        # This should call the retrieval agent's update_vectorstore_with_files
        # For now, just log
        logger.info(f"Uploaded files: {[os.path.join(UPLOAD_FOLDER, file.filename) for file in files]}")
        
        return jsonify({'success': True, 'message': 'Documents processed successfully'})
        
    except Exception as e:
        print(f"Error in upload: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_chain, chat_history, current_pdf_filename
    
    try:
        data = request.get_json()
        user_question = data.get('message', '').strip()
        
        if not user_question:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
            
        if not conversation_chain:
            return jsonify({'success': False, 'error': 'No conversation chain available. Please process documents first.'}), 400
            
        # Add user message to chat history
        chat_history.append({'role': 'user', 'content': user_question})
        
        # Query reformulation
        reformulated_query = query_reformulation_agent.process({
            "query": user_question,
            "context": "document_qa"
        })
        
        agent_id = None
        agent_name = None
        if reformulated_query["success"]:
            final_query = reformulated_query["data"]["primary_query"]
            agent_id = reformulated_query.get("agent_id", "query_reformulation")
            agent_name = get_agent_display_name(agent_id)
            chat_history.append({
                'role': 'system',
                'content': f"Query reformulated by {agent_name}",
                'agent_id': agent_id,
                'agent_name': agent_name
            })
        else:
            final_query = user_question
            
        # Check for structured data extraction
        data_type = structured_agent.detect_data_type(final_query)
        
        if data_type in ["table", "chat"] and current_pdf_filename:
            pdf_path = os.path.join("temp_uploaded_files", current_pdf_filename)
            if os.path.exists(pdf_path):
                result = structured_agent.process({
                    "pdf_path": pdf_path,
                    "data_type": data_type,
                    "pages": "all"
                })
                
                agent_id = result.get("agent_id", "structured_data_extraction")
                agent_name = get_agent_display_name(agent_id)
                if result["success"]:
                    if data_type == "table" and result["data"].get("tables"):
                        response_content = ""
                        for i, table in enumerate(result["data"]["tables"]):
                            df = pd.DataFrame(table["data"]) if isinstance(table, dict) and "data" in table else pd.DataFrame(table)
                            response_content += f'Table {i+1}:<br>{df.to_html(index=False, classes="table-auto w-full text-xs")}<br><br>'
                    elif data_type == "chat" and result["data"].get("chat_segments"):
                        response_content = ""
                        for segment in result["data"]["chat_segments"]:
                            response_content += f'<pre>{segment}</pre><br>'
                    else:
                        response_content = f'No {data_type}s could be extracted from the PDF. (Agent: {agent_name})'
                        
                    chat_history.append({'role': 'bot', 'content': response_content, 'agent_id': agent_id, 'agent_name': agent_name})
                    return jsonify({'success': True, 'response': response_content, 'agent_id': agent_id, 'agent_name': agent_name})
                else:
                    response_content = f'Failed to extract {data_type}s.'
                    chat_history.append({'role': 'bot', 'content': response_content, 'agent_id': agent_id, 'agent_name': agent_name})
                    return jsonify({'success': True, 'response': response_content, 'agent_id': agent_id, 'agent_name': agent_name})
        
        # Regular RAG processing
        retrieval_result = retrieval_agent.process({
            "queries": [final_query],
            "original_query": final_query
        })
        
        if retrieval_result["success"]:
            agent_id = retrieval_result.get("agent_id", "adaptive_retrieval")
            agent_name = get_agent_display_name(agent_id)
            chat_history.append({
                'role': 'system',
                'content': f"Query processed by {agent_name}",
                'agent_id': agent_id,
                'agent_name': agent_name
            })
        else:
            agent_id = None
            agent_name = None
        
        # Generate response
        start_time = time.time()
        response = conversation_chain({'question': final_query})
        end_time = time.time()
        fetch_time = end_time - start_time
        
        response_content = response['answer']
        if agent_id and agent_name:
            response_with_agent = f"{response_content}\n\n---\n*Response generated by {agent_name} in {fetch_time:.2f} seconds*"
            chat_history.append({'role': 'bot', 'content': response_with_agent, 'agent_id': agent_id, 'agent_name': agent_name})
            return jsonify({'success': True, 'response': response_with_agent, 'agent_id': agent_id, 'agent_name': agent_name})
        else:
            response_with_agent = f"{response_content}\n\n---\n*Response generated by NeuroFetch in {fetch_time:.2f} seconds*"
            chat_history.append({'role': 'bot', 'content': response_with_agent})
            return jsonify({'success': True, 'response': response_with_agent})
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat-history', methods=['GET'])
def get_chat_history():
    return jsonify({'success': True, 'history': chat_history})

@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    global chat_history, conversation_chain, vectorstore, current_pdf_filename
    chat_history = []
    conversation_chain = None
    vectorstore = None
    current_pdf_filename = None
    return jsonify({'success': True, 'message': 'Chat cleared successfully'})

@app.route('/agents', methods=['GET'])
def agents():
    resp = requests.get(MCP_AGENTS_URL)
    return jsonify(resp.json())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 