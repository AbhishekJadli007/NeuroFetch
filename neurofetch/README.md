# NeuroFetch

**NeuroFetch** is a full-stack, AI-powered document understanding and retrieval system. It enables users to upload and process documents (PDF, CSV, TXT, MD), extract structured data (e.g., tables, conversations), and interact via a conversational UI powered by large language models (LLMs).

---

## 🚀 Key Features

- **Modern React + Vite Frontend**: Responsive and intuitive UI for document upload and chat.
- **Node.js/Express Backend**: Manages user authentication, API endpoints, and session control.
- **Flask-based Python API**: Coordinates document parsing and LLM-driven conversational responses.
- **Multi-Agent MCP Server**: Advanced orchestration for query handling, table extraction, and intelligent routing.
- **MongoDB Integration**: Handles persistent storage for users and documents.
- **Structured Data Extraction**: Extracts tables, chats, and metadata from PDFs with tools like Camelot and pdfplumber.
- **LLM-Enhanced Q&A**: Interact with your documents using intelligent, contextual queries.

---

## 🛠️ Tech Stack

| Layer        | Technologies Used                                                                 |
|--------------|-------------------------------------------------------------------------------------|
| **Frontend** | React, Vite, Tailwind CSS, react-markdown                                          |
| **Backend**  | Node.js, Express, Mongoose, MongoDB                                                |
| **Python API** | Flask, LangChain, PyPDF2, pandas, Camelot, pdfplumber                           |
| **Utilities** | concurrently, cross-env                                                           |

---

## ⚙️ Requirements

- Node.js (v18+ recommended)
- npm
- Python 3.10+
- MongoDB (local or remote)
- *(Optional)* Ollama or other LLM backend for extended capabilities

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/neurofetch.git
cd neurofetch
```

### 2. Install Node.js Dependencies

```bash
cd neurofetch-ui
npm install
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure MongoDB

Ensure MongoDB is running locally, or update `MONGODB_URI` in `server/server.js` to point to your remote MongoDB instance.

---

## 🚀 Running the Project

### ✅ Recommended: Run Everything at Once
This starts the frontend, backend, Flask API, and MCP server together.

```bash
cd neurofetch
npm run dev:all
```

**Services:**
- Frontend: http://localhost:5173
- Node.js Backend: http://localhost:3000
- Flask API: http://localhost:5000
- MCP Server: (runs via Python, port as configured)

Refer to `package.json` for individual service scripts.

### 🔧 Run Services Individually

| Service         | Command           |
|-----------------|-------------------|
| Frontend        | npm run dev       |
| Node.js API     | npm run server    |
| Flask API       | npm run flask     |
| MCP Server      | npm run mcp       |

---

## 💡 Usage

1. Open the frontend in your browser.
2. Register or log in.
3. Upload your document (PDF, CSV, TXT, or MD).
4. Click "Process Documents".
5. Ask questions or extract tables using the chat interface.

---

## 🐍 Python Dependencies
All Python packages are listed in `requirements.txt`.

---

## 🛠 Troubleshooting
- **Port conflicts:** Ensure ports 3000, 5000, and 5173 are free.
- **Missing modules:** If you see `ModuleNotFoundError: No module named 'agents'`, run the MCP server from the `src` directory or use `npm run mcp`.
- **MongoDB errors:** Ensure MongoDB is running and URI is correctly set in the server configuration.

---

## 📜 License
MIT License — Free to use, modify, and distribute.

---

## 🙌 Acknowledgements
Inspired by modern developer experience (DX), UI/UX patterns, and open-source AI systems.

---

## 🤖 Multi-Agent MCP Server
The Multi-Agent Control and Processing (MCP) server orchestrates intelligent document understanding using specialized agents for various tasks.

### Agent Overview

| Agent Name                  | Class Name                  | Functionality                                                                 |
|-----------------------------|-----------------------------|-------------------------------------------------------------------------------|
| Query Reformulation Agent   | QueryReformulationAgent     | Enhances and expands user queries using synonyms, keywords, and intent analysis. |
| Adaptive Retrieval Agent    | AdaptiveRetrievalAgent      | Performs vector-based retrieval with chunking, embedding, deduplication, and ranking. |
| Structured Data Agent       | StructuredDataExtractionAgent | Extracts tables and conversations from documents using tools like Camelot and pdfplumber. |

### Routing Logic
- **LLM First:** Query is passed to the LLM.
- **Agent Fallback:** If LLM confidence is low, route to:
  - Tables/chats → StructuredDataExtractionAgent
  - Text retrieval → AdaptiveRetrievalAgent
- **LLM with Tools:** Output from agents is passed back to LLM for refinement if needed.

### MCP API Endpoints

**List agents**
```
GET http://localhost:8000/agents
```

**Route a query**
```
POST http://localhost:8000/route_query
```
Body:
```json
{
  "query": "Extract all tables from the attached PDF.",
  "context": {
    "pdf_path": "/path/to/file.pdf",
    "data_type": "table"
  }
}
```

**Sample Response:**
```json
{
  "agent": "structured_data_extraction",
  "response": {
    "success": true,
    "data": {
      "tables": [...],
      "extraction_method": "camelot",
      "total_tables": 3
    }
  },
  "trace": ["llm", "structured_data_extraction"],
  "elapsed": 1.23
}
```

---

## 📚 PDFBot: Streamlit Document Chat Interface
PDFBot is a lightweight, standalone Streamlit app for interactive document exploration.

### Features
- Upload and interact with PDFs, CSVs, TXTs, and MD files.
- Extract tables and chat logs.
- Uses the same agent-based logic as the main system.
- Clean UI with emoji avatars (🤖 bot, 🧑 user).
- No authentication required.

### Getting Started

**Install Python dependencies**
```bash
pip install -r requirements.txt
```

**Launch PDFBot using streamlit**
```bash
cd neurofetch-ui
streamlit run src/pdfbot.py
```

**Access it in your browser**
http://localhost:8501
