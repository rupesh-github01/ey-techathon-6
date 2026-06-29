# Gloser AI — Pharmaceutical Intelligence Platform

A full-stack AI-powered pharmaceutical market intelligence system built with a multi-agent architecture. Analyze pharmaceutical markets, clinical pipelines, patents, and trade data through a conversational interface.

## 🏗️ Architecture Overview

### Backend Components

| Component | File | Description |
|-----------|------|-------------|
| **Master Agent** | `master_agent.py` | LangGraph StateGraph workflow orchestrating all agents |
| **IQVIA Agent** | `iqvia_agent.py` | Market intelligence (market sizes, CAGR, competitors) |
| **EXIM Agent** | `exim_agent.py` | Import/export trade data for pharmaceutical APIs |
| **Patent Agent** | `patent_agent.py` | Patent landscape (filings, expiry dates, assignees) |
| **Clinical Agent** | `clinical_agent.py` | Clinical trial data (phases, sponsors, status) |
| **Flask Server** | `server.py` | RESTful API for frontend communication |

### Frontend Components

| Component | Path | Description |
|-----------|------|-------------|
| **Next.js App** | `gloser/` | Modern React chat interface |
| **Chat Hook** | `src/hooks/useChat.tsx` | State management & API calls |
| **Message Bubble** | `src/components/MessageBubble.tsx` | Message rendering with markdown |
| **Visual Renderer** | `src/components/VisualRenderer.tsx` | Charts & tables using Chart.js |
| **Sidebar** | `src/components/Sidebar.tsx` | Conversation history management |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai/) with `phi3` or `llama3` model (or configure OpenAI)

### Backend Setup

1. **Create and activate virtual environment:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. **Install dependencies:**

```powershell
pip install -r requirements.txt
```

3. **Configure environment (optional):**

```powershell
cp .env.example .env
# Edit .env with your settings
```

4. **Ensure Ollama is running with mistral model:**

```powershell
ollama pull mistral
ollama serve
```

5. **Start the Flask server:**

```powershell
python server.py
```

Server runs on `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**

```powershell
cd gloser
```

2. **Install dependencies:**

```powershell
npm install
```

3. **Start development server:**

```powershell
npm run dev
```

Frontend runs on `http://localhost:3000`

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/master-agent/query` | Run full pipeline |
| `POST` | `/api/master-agent/plan` | Get planner steps only |

**Example Request:**

```bash
curl -X POST http://localhost:8000/api/master-agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the market size for oncology drugs?"}'
```

## 🎯 Example Queries

- "What's the market size and CAGR for oncology drugs?"
- "Who are the main competitors in the diabetes market?"
- "Show clinical trials for Alzheimer's in Phase 3"
- "Find patents related to Metformin with active status"
- "What are India's top import sources for Paracetamol API?"

## 📊 Data Sources

| File | Description |
|------|-------------|
| `iqvia_data.json` | Market data for 12 therapeutic areas |
| `exim_data.json` | HS code-based trade data for APIs |
| `patent_data.json` | 132+ patent records |
| `clinical_data.json` | Sample clinical trials + live ClinicalTrials.gov API |

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **AI/ML** | LangGraph, LangChain, Ollama (mistral) |
| **Backend** | Python 3.x, Flask, Flask-CORS |
| **Frontend** | Next.js 16, React 19, TypeScript |
| **Visualization** | Chart.js, react-chartjs-2 |
| **Data** | JSON files, ClinicalTrials.gov API |

## 📁 Project Structure

```
├── master_agent.py          # Main orchestrator
├── iqvia_agent.py           # Market intelligence agent
├── exim_agent.py            # Trade data agent
├── patent_agent.py          # Patent landscape agent
├── clinical_agent.py        # Clinical trials agent
├── llm_worker.py            # LLM utility functions
├── server.py                # Flask API server
├── streamlit_app.py         # Streamlit interface (alternative)
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
├── *_data.json              # Data files
└── gloser/                  # Next.js frontend
    ├── src/
    │   ├── app/             # Next.js app router
    │   ├── components/      # React components
    │   └── hooks/           # Custom hooks
    └── package.json
```

## 🔧 Configuration

See `.env.example` for available configuration options:

- `LLM_BACKEND` - Choose between `ollama` or `openai`
- `OLLAMA_MODEL` - Default: `mistral`
- `OPENAI_API_KEY` - Required if using OpenAI
- `FLASK_PORT` - Default: `8000`

## 📝 License

MIT

# KGPharma
