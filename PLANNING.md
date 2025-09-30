## Space Biology Knowledge Engine — Hackathon Plan (NASA Space Apps 2025)

Reference: [Build a Space Biology Knowledge Engine](https://www.spaceappschallenge.org/2025/challenges/build-a-space-biology-knowledge-engine/)

### Objective
Build an end-to-end, demo-ready knowledge engine that ingests space biology documents, extracts structured insights, enables semantic search and question answering, and visualizes relationships (entities, experiments, outcomes). Optimize for reliability, speed to demo, and clear scientific value.

### Your Preferences Incorporated
- UI: Streamlit for fastest delivery and clean UX
- Compute: CPU-first pipeline with optional light GPU acceleration (RTX 3050 4GB)
- Models/APIs: Prefer open-source; allow closed-source via configurable `.env`
- Content focus: Use entire paper; give extra weight to Abstract/Introduction without over-reliance
- Retrieval: Hybrid RAG (vector + graph) with an agent choosing vector-only, graph-only, or hybrid per query
- Data source: Use current `documents/` folder for now
- Graph backend: Neo4j as the primary graph store

### LLM/Provider Strategy
- Default local model: `llama-3.1-8b-instruct` via Ollama for CPU-first usage with optional small-batch GPU
- Frontier option: OpenAI `gpt-4o-mini` / `gpt-4.1-nano` for higher-quality reasoning when toggled
- Switching: runtime-configurable via `.env` (`LLM_PROVIDER`, `LLM_MODEL`) with identical tool interfaces so the agent logic stays unchanged
- Guardrails: deterministic decoding for citation answers; temperature scheduling (low for fact QA, higher for summaries)

### What to Build (Scope and Deliverables)
- **Ingestion + Parsing**: Batch import of PDFs in `documents/`, robust text extraction (incl. tables, figures’ captions), metadata capture (title, authors, year, DOI if available).
- **Knowledge Extraction**: NER for biological entities (organism, tissue, gene/protein, experimental condition, spaceflight variable), relation/event extraction (e.g., “spaceflight → gene X upregulated in tissue Y”).
- **Indexing & Search**: Hybrid search (BM25 + embeddings) over passages; filters by metadata; top-k contexts for Q&A.
- **Q&A (RAG)**: Grounded answers with citation snippets and confidence; show exact passages used.
- **Knowledge Graph**: Entity and relation graph from extracted triples; interactive visualization and entity drill-down.
- **Simple Web UI (Streamlit)**: One-page app with: search bar, Q&A panel with sources, left filters, right KG visualization; optional admin-only upload/ingest; caching for low latency.
- **API**: `/ingest`, `/search`, `/answer`, `/graph` endpoints for reproducibility and integration.
- **Reproducibility/Docs**: One-command setup, seeded demo dataset, evaluation notebook, and a concise demo script.

### High-Level Architecture
- **Data Layer**
  - Storage: local corpus dir (`documents/`) + parsed JSONL store
  - Vector index: FAISS/Chroma (local, hackathon-friendly)
  - Graph: Neo4j (primary, supports AuraDB); in-memory NetworkX fallback with persistence
- **Services**
  - Ingestion worker: parse PDFs, run OCR fallback, normalize metadata
  - NLP pipeline: NER, relation extraction, chunking, embeddings
  - Search/Q&A: hybrid retrieval → rerank → RAG answer with citations; agent selects vector/graph/hybrid tool per query
  - Graph builder: accumulate triples, deduplicate, persist
  - Scheduler/Queue: background jobs for ingest and re-index to handle scale (e.g., Celery/RQ)
- **UI**
  - Streamlit single-page app with: search, answers, filters, KG; server-side caching; optional admin-only ingest view

### Execution Flow (End-to-End)
1) User drags PDFs into the app (or uses preloaded corpus) → ingestion queue
2) Parser extracts text, sections, tables (where possible), metadata → normalized JSONL
3) Chunking (semantic/section-aware); embeddings computed; hybrid index updated
4) NER + relation extraction over chunks; triples accumulated → graph store
5) User query → retrieve (BM25 + dense) and/or Neo4j graph traversal → rerank → assemble context → LLM answer
6) UI displays answer, confidence, citations (with page/section), and highlights nodes/edges in KG related to the query

### MVP First, Then Plus-Ones
- **MVP (Day 1)**
  - Parsing for most PDFs (PyMuPDF); OCR fallback (Tesseract) if needed
  - Chunking + embeddings; BM25 + dense retrieval; simple RAG with citations
  - Minimal NER (organism, tissue, condition) with off-the-shelf model; naive triple extraction templates
  - Streamlit UI: search box, answers with citations, basic KG visualization; ingest hidden behind an admin toggle
- **Polish (Day 2)**
  - Better NER (SciSpaCy/BioBERT), relation patterns for spaceflight variables (microgravity, radiation)
  - Graph deduplication/merging; entity pages (e.g., organism view with linked papers)
  - Reranking (cross-encoder), section-aware chunking with Abstract/Intro upweighting (not exclusive), table caption inclusion
  - Evaluation harness and demo script; export sharable findings (PDF/CSV)

### Agentic Hybrid RAG (Vector + Graph)
- Tools available to the agent:
  - `retrieve_vector(query, k, filters)`: dense + BM25 passage retrieval
  - `retrieve_graph(entity|pattern)`: Neo4j Cypher templates (entity lookup, relation by type, path expansions)
  - `retrieve_hybrid(query)`: align vector contexts with subgraphs around matched entities
  - `rerank(passages, query)`: CPU-friendly cross-encoder reranker
  - `answer_with_citations(contexts, query)`: constrained generation with exact citations
- Decision policy: classify queries into overview/definition (vector), entity/relationship (graph), causal/comparative (hybrid). Back off to hybrid if confidence is low.
- Latency targets: <2.0s P50, <4.0s P95 on CPU; cache embeddings, retrieval results, and Neo4j subgraphs.

### Proposed Tech Choices (hackathon-optimized)
- **Parsing**: PyMuPDF; `pytesseract` OCR fallback; optional GROBID for metadata
- **NLP**: spaCy + SciSpaCy for NER; rule/LLM-assisted relation extraction
- **Embeddings (CPU-first)**: `bge-small-en` or `all-MiniLM-L6-v2` (optionally GPU-accelerated for batches)
- **Reranker (CPU-friendly)**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Index**: FAISS or Chroma; BM25 via `rank-bm25` or similar
- **RAG**: Prefer open-source (e.g., `ollama`/`vLLM` with small instruct models) with API fallback (OpenRouter/OpenAI/Azure); strict grounding and citation enforcement
- **Graph**: Neo4j (supports AuraDB) for storage and Cypher querying; visualization via `pyvis` or `streamlit-graphviz`
- **UI**: Streamlit (chosen)

### Data Model (concise)
- Document: id, title, authors, year, venue, doi, sections, chunks
- Chunk: id, doc_id, text, section, page, embeddings, entities
- Entity: id, type (organism, tissue, gene, condition, variable)
- Relation: subject → predicate → object, provenance (doc_id, chunk_id, span)

### Quality & Evaluation
- Retrieval: Recall@k on a small set of handcrafted Q/A based on the corpus
- Answer Quality: Human-rated groundedness, citation correctness, hallucination rate
- Extraction: Spot-check triples against source sentences; report precision
- Performance: Ingest throughput, search latency; UI responsiveness

### Risk Mitigation
- PDF variety: use robust parser + OCR fallback; skip damaged docs with clear logs
- Hallucinations: enforce quoted citations, show used contexts, abstain on low confidence
- Time constraints: MVP first; keep advanced features feature-flagged

### Configuration (.env) Plan
- Flexible provider selection and keys:
  - `LLM_PROVIDER` = `openrouter|openai|azure|ollama|vllm|none`
  - `LLM_MODEL` = default small instruct model name
  - `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, etc.
  - `EMBEDDING_MODEL` = `bge-small-en` (default)
  - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (works with Neo4j AuraDB)
  - `USE_GPU` = `true|false` (default false)
  - `MAX_CONTEXT_TOKENS`, `RETRIEVAL_TOP_K`, `RERANK_TOP_K`
  - Safe defaults with environment-driven overrides

### Testing & Evaluation Plan
- Automated tests (post-build):
  - Retrieval: Recall@k and MRR over a small labeled set (query → gold passages)
  - QA Grounding: verify that every claim in the answer maps to cited spans; fail if uncited
  - Hallucination checks: adversarial prompts and negative controls; enforce abstention when low confidence
  - Consistency: paraphrase queries; expect stable answers and citations overlap
  - Latency: P50/P95 timing for ingest, retrieval, rerank, answer; budget alerts
- Scripts/Artifacts:
  - `tests/test_retrieval.py`, `tests/test_grounding.py`, `tests/test_latency.py`
  - `notebooks/eval_report.ipynb` summarizing metrics with charts
  - Red-team prompt list in `tests/prompts/adversarial.txt`

### Demo Narrative (judge-focused)
1) Pose a realistic science question (e.g., “How does microgravity affect muscle atrophy in mice?”)
2) Show answer with 2–3 citations, highlight the exact sentences used
3) Click to graph: show relationships (Organism: Mouse → Tissue: Muscle → Condition: Microgravity → Outcome: Atrophy)
4) Filter papers by organism/year to reveal trends; export findings
5) Briefly reveal architecture slide and evaluation metrics; end with impact statement

### Team Roles (suggested)
- Lead: keeps scope on track; handles demo
- Data/Parsing: ingestion reliability, metadata normalization
- NLP/ML: NER, relation extraction, embeddings, rerankers
- Backend: indexing, RAG endpoints, graph builder
- Frontend: UI/UX, graph viz, polish

### Setup & Repro (hackathon-ready)
- One command to run: `uv run app.py` or `docker compose up`
- Seeded with a small subset of space biology PDFs for instant demo
- `.env.example` for keys and provider selection; offline fallback models when possible

### Project Structure (proposed)
- `app/` — FastAPI or lightweight service layer exposing `/ingest`, `/search`, `/answer`, `/graph`
- `ui_streamlit/` — Streamlit app (pages: Search/Q&A, Graph, Ingest)
- `ingestion/` — PDF parsing, OCR fallback, metadata normalization
- `nlp/` — NER, relation extraction, section-aware chunking
- `rag/` — retrieval, reranking, agent/tool orchestration, prompt templates
- `graph/` — Neo4j connectors, Cypher templates, graph builder/deduper
- `index/` — FAISS/Chroma index management and embeddings
- `config/` — `.env` loader, provider switch logic, constants
- `tests/` — automated tests and evaluation harness
- `notebooks/` — exploratory and evaluation notebooks
- `scripts/` — CLI for batch ingest, rebuild index, export graph, run eval
- `data/` — intermediate caches and outputs (excluded from git if large)
- `documents/` — raw PDFs (already present)

### Success Criteria (what wins)
- Clear scientific value: trustworthy, cited, and useful insights for researchers
- Smooth UX: fast answers, readable citations, intuitive graph
- Reproducible build + public repo; concise, compelling demo story
- Ethical data use and transparency (provenance, limitations)

---

### Optional Clarifications
1) Preferred small open-source instruct model to start with (e.g., `mistral-7b-instruct` vs. `llama-3.1-8b-instruct`), given RTX 3050 constraints?
2) Acceptable external services for demo (e.g., managed Neo4j Aura) or strictly local?
3) Target demo length and 2–3 headline questions you want to showcase?

---

## Agent-Ready Build Specification

The following sections provide precise contracts and step-by-step milestones so any LLM/Agent can implement this project end-to-end without ambiguity.

### Implementation Milestones & Checklist
1) Repository scaffold
   - Create folders: `app/`, `ui_streamlit/`, `ingestion/`, `nlp/`, `rag/`, `graph/`, `index/`, `config/`, `tests/`, `notebooks/`, `scripts/`, `data/`
   - Add `.env.example` with all variables from Configuration Plan
   - Add `pyproject.toml` dependencies (see Dependencies) and `README.md` quickstart
2) Ingestion & Parsing
   - Implement PDF loader with PyMuPDF; OCR fallback via Tesseract
   - Normalize metadata (title, authors, year, doi); persist parsed JSONL in `data/parsed/`
   - Section-aware chunking; store chunks with `doc_id`, `section`, `page`
3) Indexing
   - Build embeddings (CPU-first model); create FAISS/Chroma index in `data/index/`
   - BM25 text index (rank-bm25)
   - Background job to re-index when new docs added
4) NLP Extraction
   - NER with spaCy/SciSpaCy; map to entity schema
   - Rule/LLM-assisted relation extraction; store triples with provenance
5) Graph
   - Neo4j connectors; create constraints/indexes; upsert entities/relations
   - Graph builder job from extracted triples
6) Retrieval & Rerank
   - Vector + BM25 retrieval; CPU-friendly cross-encoder reranker
   - Graph retrieval via parameterized Cypher templates
7) Agentic RAG
   - Implement tool interfaces; decision policy for vector/graph/hybrid
   - Grounded generation with citation enforcement
8) API Layer
   - Implement `/ingest`, `/search`, `/answer`, `/graph` per contracts below
9) Streamlit UI
   - Search/Q&A pane with citations; filters; KG visualization; admin ingest page (toggle)
10) Testing & Evaluation
   - Implement tests and evaluation scripts; generate basic eval report
11) Packaging & Demo
   - One-command run; sample corpus; short demo script

### API Contract (HTTP)
- `POST /ingest`
  - body: `{ paths: string[] | null }` (null = use `documents/`)
  - resp: `{ accepted: number, failed: number, job_id: string }`
- `GET /search`
  - query: `q`, `k=10`, `filters?`
  - resp: `{ query: string, results: [ { doc_id, chunk_id, score, text, section, page } ] }`
- `POST /answer`
  - body: `{ query: string, mode?: "auto"|"vector"|"graph"|"hybrid", k?: number }`
  - resp: `{ answer: string, citations: [ { doc_id, chunk_id, text, section, page } ], reasoning?: string, confidence: number }`
- `GET /graph/entity`
  - query: `name`, `type?`
  - resp: `{ entity: { id, name, type }, neighbors: [ { id, name, type, relation } ] }`
- `GET /graph/path`
  - query: `source`, `target`, `max_len=3`
  - resp: `{ nodes: [...], edges: [...] }`

### Data Schemas (JSONL / internal)
- Document
  - `{ doc_id, title, authors: string[], year: number|null, doi: string|null, sections: [ { name, text, page_start, page_end } ] }`
- Chunk
  - `{ chunk_id, doc_id, section, page, text, embedding: float[]|null, entities: EntityRef[] }`
- EntityRef
  - `{ span: [start,end], text, type: "organism"|"tissue"|"gene"|"condition"|"variable" }`
- Triple
  - `{ subject: Entity, predicate: string, object: Entity, provenance: { doc_id, chunk_id, span: [start,end] } }`

### LLM Tooling Contract (Agent Interfaces)
- `retrieve_vector(query: string, k: int, filters?: dict): Passage[]`
- `retrieve_graph(entityOrPattern: string|dict): GraphResult`
- `retrieve_hybrid(query: string): { passages: Passage[], subgraph: GraphResult }`
- `rerank(passages: Passage[], query: string, top_k: int): Passage[]`
- `answer_with_citations(contexts: Passage[], query: string, max_tokens?: int): { answer: string, citations: Passage[] }`
Notes:
- All tools are pure; no side-effects. Time budget per call should target < 800 ms on CPU where possible.
- The agent must abstain when grounding is insufficient and ask for refinement.

### Environment Variables Reference (.env.example)
- `LLM_PROVIDER=openai|ollama|openrouter|azure|vllm|none`
- `LLM_MODEL=llama-3.1-8b-instruct`
- `OPENAI_API_KEY=...`
- `OPENROUTER_API_KEY=...`
- `AZURE_OPENAI_API_KEY=...`
- `EMBEDDING_MODEL=bge-small-en`
- `NEO4J_URI=neo4j+s://<aura-host>`
- `NEO4J_USER=neo4j`
- `NEO4J_PASSWORD=...`
- `USE_GPU=false`
- `MAX_CONTEXT_TOKENS=4096`
- `RETRIEVAL_TOP_K=10`
- `RERANK_TOP_K=5`

### Dependencies (Python)
- Core: `fastapi`, `uvicorn`, `pydantic`
- Streamlit: `streamlit`
- Parsing/OCR: `pymupdf`, `pytesseract`, `Pillow`
- NLP: `spacy`, `scispacy`, `sentence-transformers`, `rank-bm25`
- Vector DB: `faiss-cpu` or `chromadb`
- Graph: `neo4j`
- Eval/Utils: `numpy`, `pandas`, `scikit-learn`, `tqdm`, `pyvis`
- Testing: `pytest`

### Runbook
- Dev
  - `cp .env.example .env` and fill keys
  - `uv run app.py` to start API; `streamlit run ui_streamlit/App.py` to start UI
  - Place PDFs in `documents/`; call `/ingest` or use admin page
- Production/Demo
  - Use Neo4j AuraDB; keep FAISS/Chroma on local storage or attach volume
  - Prebuild indexes before demo; warm caches with top demo queries

### Evaluation Protocol (Automation)
- Run: `uv run scripts/run_eval.py --queries tests/queries.jsonl --gold tests/gold.jsonl`
- Outputs: `notebooks/eval_report.ipynb` containing Recall@k, grounding accuracy, hallucination rate, and latency histograms
- Thresholds to pass:
  - Recall@10 ≥ 0.75 on demo set
  - Grounding coverage ≥ 0.95 of claims cited
  - Hallucination rate ≤ 5%
  - P95 end-to-end answer latency ≤ 4s (CPU)

### Non-Functional Requirements
- Scalability: background job queue; incremental indexing; idempotent upserts to Neo4j
- Observability: basic request and job logs; latency metrics per stage
- Security: admin-only ingest; secrets via `.env`; no keys in repo
- Portability: local-first; cloud-ready with AuraDB and API provider toggles

### File/Folder Responsibilities (Concise)
- `app/`: routers, request/response models, dependency injection, error handling
- `ui_streamlit/`: pages, UI state, caching, API client
- `ingestion/`: loaders, OCR, normalization, JSONL writer
- `nlp/`: spaCy pipelines, relation heuristics, section parser
- `index/`: embedding generator, FAISS/Chroma builders, BM25 index
- `graph/`: Neo4j client, Cypher templates, upsert utilities
- `rag/`: retrieval orchestration, reranker, agent policy, prompts
- `config/`: env loader, provider selection, constants
- `tests/`: retrieval/grounding/latency tests and fixtures
- `scripts/`: CLI utilities for batch operations and evaluation

### Acceptance Criteria (for agents)
- All API endpoints implemented and documented
- Ingested sample corpus indexed; at least 200 chunks with embeddings
- Neo4j populated with ≥ 200 entities and ≥ 300 relations
- Streamlit UI can answer queries with citations; graph panel interactive
- Tests pass with thresholds specified in Evaluation Protocol
- `.env.example` complete; README has 5-minute quickstart


