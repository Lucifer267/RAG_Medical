# 🏥 Real-Time Hybrid Graph RAG: Medical Question-Answering System

> **An advanced real-time medical QA system that compares three distinct retrieval approaches and autonomously selects the most factually accurate answer using LLM judgment.**

---

## 🎯 Executive Summary

**Real-Time Hybrid Graph RAG** is a sophisticated medical question-answering system that demonstrates the power of integrating multiple AI approaches. Rather than relying on a single retrieval method, this system executes three different strategies in parallel:

1. **Pure LLM**: Direct knowledge from a language model
2. **Vector RAG**: Semantic search with Chroma embeddings
3. **Hybrid Graph RAG**: Combined vector + knowledge graph context

An LLM "judge" automatically evaluates all three responses and selects the most medically accurate answer in real-time, while an interactive dashboard visualizes performance metrics and historical trends.

**Key Innovation**: This comparative approach systematically reveals when specialized retrieval methods outperform others, making it both a powerful medical QA tool and an excellent educational framework for understanding RAG architectures.

---

## ✨ Key Features

### 🔬 **Triple-Mode Architecture**
- **LLM-Only Mode**: Baseline pure model performance
- **Vector Retrieval Mode**: Semantic similarity from FDA drug labels and MedlinePlus topics
- **Hybrid Graph Mode**: Knowledge graphs + vector context for structured reasoning

### 🏆 **Intelligent Auto-Judging**
- LLM automatically compares medical fact density across all three answers
- Objective evaluation criteria prioritize accuracy over writing style
- Real-time performance metrics and response timing

### 📊 **Interactive Dashboard**
- Real-time result visualization with 3-card comparative layout
- Historical query tracking with `localStorage`
- Win statistics dashboard (doughnut chart showing mode success rates)
- Response time analysis with bar/radar charts

### 📚 **Medical Data Integration**
- **FDA DailyMed**: 5,000+ drug labels with detailed pharmaceutical information
- **MedlinePlus**: Health topics, conditions, and treatments with hierarchical organization
- Local SQLite-based Chroma collections for persistent embeddings
- Neo4j graph structure for entity relationships and medical taxonomies

### ⚡ **Production-Ready Backend**
- Python Flask REST API with CORS support
- Local Ollama inference (Llama3 model) - no API costs, full privacy
- Parallel execution engine for identical response latency
- Thread-safe database connections

---

## 🏗️ Architecture Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  FDA DailyMed XML           MedlinePlus Health Topics            │
│  (Drug labels)              (Conditions & Treatments)            │
│       │                              │                           │
│       └──────────────────┬───────────┘                           │
│                          ↓                                        │
│                  ┌─────────────────┐                             │
│                  │  Data Parsing   │                             │
│                  │  & Extraction   │                             │
│                  └────────┬────────┘                             │
│                           │                                       │
│          ┌────────────────┼────────────────┐                     │
│          ↓                ↓                ↓                     │
│    [Neo4j Graph]   [Chroma Vector DB]  [LLM Enrichment]        │
│    • Drug nodes    • Embeddings        • Entity extraction      │
│    • Topic nodes   • Semantic search   • Relationship mining    │
│    • Relationships • Persistent SQLite • Knowledge enrichment   │
│    • Taxonomies    • k-NN retrieval                            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │   User Query    │
                    └────────┬────────┘
                             │
                 ┌───────────┼───────────┐
                 │           │           │
                 ↓           ↓           ↓
              Mode A       Mode B       Mode C
              (LLM)      (Vector)      (Hybrid)
              │           │             │
              └───────────┼─────────────┘
                          ↓
                    [LLM Judge]
                  (Fact Evaluation)
                          ↓
              ┌─────────────────────┐
              │  Best Answer + All 3│
              │  Timing & Stats     │
              └─────────────────────┘
                          ↓
                   [Web Dashboard]
                  (Real-time Display)
```

### System Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend API** | Flask + Flask-CORS | RESTful endpoints for QA and data management |
| **Vector DB** | Chroma + SQLite | Persistent semantic embeddings (5,000+ docs) |
| **Graph DB** | Neo4j | Structured relationships and entity hierarchies |
| **LLM Engine** | Ollama + Llama3 | Local inference, medical reasoning, auto-judging |
| **Frontend** | HTML5 + Chart.js | Interactive dashboard with real-time visualization |
| **Embeddings** | SentenceTransformers | `all-MiniLM-L6-v2` model for semantic search |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Neo4j** (running at `bolt://localhost:7687`)
- **Ollama** with Llama3 model installed
- **Git** for cloning and version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/real-time-hybrid-graph-rag.git
   cd real-time-hybrid-graph-rag
   ```

2. **Create a Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Neo4j connection**
   
   Update the credentials in `qa_full.py`, `ingest_dailymed.py`, and `enrich_graph_llm.py`:
   ```python
   NEO4J_URI = "bolt://localhost:7687"
   NEO4J_USER = "neo4j"
   NEO4J_PASSWORD = "your_password"
   ```

5. **Verify Ollama is running**
   ```bash
   ollama serve  # In another terminal
   ```

### Data Ingestion

**Option 1: Ingest FDA DailyMed Data**
```bash
python ingest_dailymed.py
# Processes all XML files in data/Dailymed\ data/
# Creates Drug nodes in Neo4j + embeddings in Chroma
```

**Option 2: Ingest MedlinePlus Health Topics**
```bash
python ingest_medlineplus.py
# Parses health_topics.xml
# Creates Topic, TopicGroup, and MeshDescriptor nodes
```

**Option 3: Enrich Graph with LLM Extractions**
```bash
python enrich_graph_llm.py
# Extracts entities and relationships from documents
# Enriches Neo4j with LLM-derived knowledge
```

### Launch the Application

```bash
python server.py
# Server starts at http://localhost:5000
# Open browser and access the dashboard
```

---

## 💡 How It Works

### Request Flow

1. **User submits a medical question** via the web interface
2. **Server initiates 3 parallel processes**:
   - **LLM-Only**: Direct model inference
   - **Vector Retrieval**: Chroma semantic search (top-5 docs) + LLM generation
   - **Hybrid Mode**: Vector search + Neo4j graph context + LLM generation

3. **LLM Judge evaluates all 3 responses**:
   - Ranks by medical fact density
   - Considers accuracy over writing style
   - Returns winner determination

4. **Dashboard displays results**:
   - All 3 answers with response times
   - Highlighted "best" response
   - Updated win statistics
   - Saved to history

### Code Organization

```
├── server.py                 # Flask API + frontend serving
├── qa_full.py               # Core QA logic (3 modes + judge)
├── chroma.py                # Chroma collection utilities
├── enrich_graph_llm.py       # LLM-based graph enrichment
├── ingest_dailymed.py        # FDA drug label ingestion
├── ingest_medlineplus.py     # MedlinePlus topic ingestion
├── index.html               # Web dashboard
├── static/
│   ├── script.js            # Frontend logic & API calls
│   ├── style.css            # Responsive styling (3 modes)
│   └── libs/                # Third-party libraries
├── data/
│   ├── Dailymed\ data/      # FDA XML files (~5,000 documents)
│   ├── dailymed_sample/     # Sample subset
│   └── medilineplus/        # Health topics XML
├── chroma_healthcare/       # Persistent Chroma vectors
└── README.md                # This file
```

---

## 📋 API Endpoints

### `POST /api/ask`
Executes the triple-mode RAG pipeline and returns all 3 answers with auto-judging.

**Request**:
```json
{
  "question": "What are the side effects of Lisinopril?"
}
```

**Response**:
```json
{
  "llm": {
    "text": "Lisinopril may cause dizziness, fatigue...",
    "time": 4.5
  },
  "vector": {
    "text": "According to FDA labels, common side effects include...",
    "time": 12.3
  },
  "hybrid": {
    "text": "Lisinopril (ACE inhibitor) side effects: dizziness, cough...",
    "time": 18.7
  },
  "winner_letter": "C",
  "winner_label": "Hybrid (Best)"
}
```

### `GET /`
Serves the interactive web dashboard.

---

## 🎨 Frontend Features

### Dashboard Components

- **Query Input**: Medical question text field with real-time validation
- **3-Card Results Layout**: 
  - 🧠 **LLM-Only** (Orange theme)
  - 📚 **Vector RAG** (Blue theme)
  - 🔗 **Hybrid RAG** (Purple theme)
- **Statistics Panel**:
  - Win distribution doughnut chart
  - Response time comparison chart
  - Historical query list
- **Responsive Design**: Mobile-friendly layout with fixed sidebar

### Technologies
- **HTML5**: Semantic markup
- **CSS3**: Flexbox layout, custom variables for theme consistency
- **Chart.js**: Real-time visualization (doughnut, bar, radar charts)
- **Vanilla JavaScript**: No framework dependencies (minimal bundle size)
- **localStorage**: Client-side history and statistics persistence

---

## 🔧 Configuration & Customization

### Database Credentials
Located in Python files (consider using environment variables for production):
```python
# NEO4J
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Maithili"

# CHROMA
CHROMA_DIR = "chroma_healthcare"
CHROMA_COLLECTION = "healthcare_topics"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# OLLAMA
LLM_MODEL = "llama3"
LLM_TIMEOUT_JUDGE = 120  # seconds
```

### Tuning Parameters

**Vector Retrieval**:
- `k=5` (top-k documents retrieved)
- Embedding model: `all-MiniLM-L6-v2` (384-dimensional vectors)

**Graph Building**:
- Neo4j Cypher queries filter by entity types and relationships
- Configurable through `build_graph_context()` in `qa_full.py`

**LLM Judging**:
- Factual accuracy prioritized in evaluation prompt
- Timeout: 120 seconds
- Temperature: Default (configured in Ollama)

---

## 📊 Performance Characteristics

### Measured Metrics

- **LLM-Only Response**: ~5-10 seconds
- **Vector Retrieval**: ~10-15 seconds (Chroma k-NN + LLM generation)
- **Hybrid Mode**: ~15-25 seconds (vector + graph + LLM)
- **Judge Evaluation**: ~15-30 seconds
- **End-to-End**: ~30-60 seconds total (3 modes + judge in sequence)

### Optimization Opportunities

- Implement caching for frequently asked questions
- Use async/await for concurrent database queries
- Batch graph database queries
- Fine-tune embedding model for medical domain
- Implement response streaming for better UX

---

## 🎓 Educational Value

This project teaches:

1. **RAG Architecture**: Multiple retrieval strategies and their trade-offs
2. **Vector Databases**: Semantic search with Chroma and embeddings
3. **Graph Databases**: Structured knowledge representation with Neo4j
4. **LLM Integration**: Local inference with Ollama and prompt engineering
5. **Web Development**: Flask REST APIs, real-time dashboards, localStorage
6. **Data Processing**: XML parsing, medical data enrichment, entity extraction
7. **Full-Stack Engineering**: Backend logic, frontend visualization, database design

---

## 🌟 Highlights for Resume

### Technical Achievements
✅ **Hybrid Architecture**: Combines vector + graph database approaches, demonstrating deep understanding of modern information retrieval  
✅ **Comparative Analysis**: LLM-based evaluation framework autonomously selects best-performing method  
✅ **Real-Time System**: Parallel execution of three QA modes with interactive visualization  
✅ **Production Patterns**: REST API, CORS, error handling, persistent storage, database indexing  
✅ **Full-Stack Implementation**: Backend (Python/Flask), Frontend (HTML/CSS/JS), Database (Neo4j/Chroma)  
✅ **Medical Data Processing**: FDA XML parsing, MedlinePlus integration, entity-relationship extraction  

### Technologies Demonstrated
- **Languages**: Python, JavaScript, HTML/CSS, Cypher
- **Backends**: Flask, Ollama
- **Databases**: Neo4j, Chroma (SQLite), Embeddings
- **Libraries**: LangChain, sentence-transformers, Chart.js, lxml
- **Concepts**: RAG, Knowledge Graphs, Vector Search, LLM Judgment, Real-Time Dashboards

### Scalability & Best Practices
- Modular code organization (separation of concerns)
- Database connection pooling with Neo4j driver
- Persistent caching with Chroma SQLite backend
- RESTful API design with JSON responses
- Frontend state management with localStorage
- Error handling and graceful degradation

---

## 🔐 Security Considerations

For production deployment:
- ✅ Move credentials to environment variables (`.env` file with `python-dotenv`)
- ✅ Implement authentication (JWT tokens for API access)
- ✅ Use HTTPS/SSL for data in transit
- ✅ Add rate limiting to prevent API abuse
- ✅ Validate and sanitize user input prompts
- ✅ Implement audit logging for medical data access (HIPAA compliance if handling real patient data)
- ✅ Use Neo4j's built-in authentication instead of hardcoded credentials

---

## 📈 Future Enhancements

1. **Advanced Filtering**: Filter results by drug class, condition, or data source
2. **Explainability**: Show which retrieved documents influenced each answer
3. **User Feedback Loop**: Let users rate answer quality to train a local ranker
4. **Multi-Model Inference**: Compare different LLM models (Mistral, Med-LLaMA, etc.)
5. **Streaming Responses**: Server-sent events for real-time answer generation
6. **Medical NER**: Named entity recognition for extracting medical terms from queries
7. **Citation Tracking**: Link answers to source documents with confidence scores
8. **Database Optimization**: Vector indexing (HNSW) for sub-second retrieval
9. **PDF Ingestion**: Support for research papers and medical documents beyond structured data

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit changes with descriptive messages
3. Push to your fork
4. Open a Pull Request with detailed description

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Contact & Support

For questions or issues:
- Open an issue on GitHub
- Contact: [Your Email/Website]
- Documentation: Full code comments and docstrings included

---

## 🙏 Acknowledgments

- **Medical Data**: FDA DailyMed and MedlinePlus for comprehensive drug/health information
- **LLM**: Meta's Llama3 model via Ollama for local, privacy-preserving inference
- **Vector Embeddings**: Hugging Face's SentenceTransformers project
- **Frameworks**: Flask, Neo4j, Chroma, Chart.js communities

---

<div align="center">

**Built with ❤️ for advancing medical information retrieval through AI**

⭐ If you find this project useful, please consider giving it a star!

</div>
