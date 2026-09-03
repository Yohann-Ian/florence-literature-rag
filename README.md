# Florence

A retrieval-augmented generation (RAG) agent that answers questions about a small corpus of literary works. Built with LangGraph, ChromaDB, and the Anthropic API. Deployed as a containerised FastAPI service.

**Currently indexed:** *Nineteen Eighty-Four* (George Orwell), *Brave New World* (Aldous Huxley). Initial corpus had more books, but I constrained it to two books for my personal demo, so it was more managable.

**UI preview (static, no backend):** [florence-ui-preview.netlify.app](https://florence-ui-preview.netlify.app/)

The UI preview lets you click through the interface as it was designed, but there is no live agent behind it. To run Florence against the corpus, clone and follow the quick start below.

---

## What Florence does

Given a literary question, Florence:

1. Checks whether the question is within her domain (Haiku classifier + prompt-injection filter).
2. Retrieves the top-5 most relevant child chunks from a local ChromaDB vector store.
3. Grades each chunk for relevance in parallel (five concurrent Haiku calls).
4. If fewer than 2 chunks pass, rewrites the query (Sonnet) and retries retrieval, capped at 2 rewrites.
5. Generates an answer grounded in the surviving chunks (Sonnet).
6. Audits the answer against those chunks using LLM-as-judge, producing faithfulness and confidence scores.
7. If confidence is low, pauses the graph and asks a human to approve or reject with feedback. Human rejection re-triggers generation with feedback injected into the prompt, capped at 3 rejections.

The pipeline is implemented as a LangGraph state machine with a MemorySaver checkpointer so the human-in-the-loop pause can survive stateless HTTP requests.

---

## Quick start

**Requirements:** Python 3.12, ~5 GB free disk (for the BGE embedding model + ChromaDB index), an Anthropic API key.

```bash
git clone https://github.com/Yohann-Ian/florence-literature-rag.git
cd florence-literature-rag

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set your API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# Build the ChromaDB index (one-time, ~10 minutes on GPU, ~30 on CPU)
python ingestion/embed_index.py

# Run the API
python -u -m uvicorn api.main:app --port 8000 --log-level info
```

Then open `http://127.0.0.1:8000/` in a browser.

To run the evaluation harness:

```bash
python evaluation/ragas_eval.py
```

Results are printed to the terminal and written as an HTML report in `evaluation/reports/`.

---

## Architecture

### Ingestion (offline, one-time)

Books are parsed from EPUB or PDF (EbookLib for EPUB, PyMuPDF for PDF), cleaned to strip Gutenberg boilerplate and normalise whitespace, then chunked hierarchically:

- **Parent chunks:** 1024 characters, 128-character overlap
- **Child chunks:** 256 characters, 32-character overlap

Each child stores a metadata pointer back to its parent. Retrieval hits children (for search precision), then fetches the corresponding parent (for LLM context).

Chunks are embedded using `BAAI/bge-large-en-v1.5` (1024-dimensional) and stored in two persistent ChromaDB collections (one for children, one for parents) using cosine similarity with HNSW indexing.

### Agent (per-request state machine)

Implemented in LangGraph. State is a `TypedDict` with fields for query, retrieved docs, grades, answer, scores, confidence, and human feedback. Routing between nodes is conditional on signal constants (`SUFFICIENT`, `INSUFFICIENT`, `EXHAUSTED`, `GROUNDED`, `UNGROUNDED`, `UNGROUNDED_FINAL`).

Model choice per node:

| Node | Model | Rationale |
|---|---|---|
| `domain_router` | Haiku | Yes/no classifier at ten tokens; Using Opus would be ridiculous. Using Fable would be plain funny. |
| `grade_documents` | Haiku (x5 parallel) | Parallel execution via `ThreadPoolExecutor` gives ~10x speedup over sequential Opus. |
| `rewrite_query` | Sonnet | Needs to rephrase intelligently; Haiku produces weak rewrites. |
| `generate` | Sonnet, temp 0.3, max_tokens 4096 | Balance of quality and cost. Opus was too expensive for the marginal gain. |
| `check_hallucination` | Sonnet, temp 0, JSON output | Haiku was tried first but produced unreliable JSON. |

### Serving

FastAPI service exposing:

- `GET /` serves the static frontend from `api/static/index.html`
- `GET /health` health check for the load balancer
- `POST /query` submit a question; returns answer, pause, or rejection
- `POST /resume` feed human decision back into a paused graph (uses LangGraph's `Command(resume=...)`)

Pydantic validates all requests and responses. Frontend is plain HTML, CSS, and vanilla JavaScript, no build pipeline.

### Deployment

Containerised via Docker:

- Base: `python:3.12-slim`
- CPU-only PyTorch (installed explicitly from `download.pytorch.org/whl/cpu` to avoid pulling gigabytes of unused CUDA libraries)
- BGE-large model pre-downloaded at build time as its own layer
- ChromaDB index baked into the image
- Final image ~4 GB

Deployed to AWS: pushed to ECR, run on ECS Fargate Express Mode (1 vCPU, 2 GB RAM), behind an Application Load Balancer. Logs stream to CloudWatch. API key injected at runtime via environment variable, never baked into the image.

Note: the live deployment is not currently running to avoid ongoing costs. The Docker image and deployment scripts remain in the repo; recreating the service takes ~10 minutes.

---

## Evaluation

A gold set of 20 questions lives in `evaluation/gold_set.json`, split into three tiers:

- **single_text**: questions answerable from one book alone
- **cross_text_thematic**: questions requiring comparison across books
- **adversarial**: out-of-domain questions that should be rejected outright

The evaluation harness (`evaluation/ragas_eval.py`) runs the 17 answerable questions through the full Florence graph, then scores each via Ragas using three reference-free metrics:

- **Faithfulness**: are the answer's claims grounded in the retrieved chunks?
- **Answer Relevancy**: does the answer address the question?
- **Context Precision**: were the retrieved chunks actually relevant?

Judge model: `claude-opus-4-5` at temperature 0 for reproducibility.

The 3 adversarial questions get a separate behavioural pass/fail check (did Florence reject them as out-of-domain?).

Results are written to a side-by-side HTML report in `evaluation/reports/` placing Florence's answer next to a model-drafted reference answer. Reference answers are for qualitative comparison only; they are not passed to Ragas and do not influence the scores.

**Evaluation caveat:** the current numbers were produced during broader-scope development when the corpus included ten books. Since scoping down to two, evaluation has not been re-run at the reduced scope. The methodology is stable; the numbers should be understood as characteristic of the broader system.

---

## Repository structure

```
florence-literature-rag/
├── agent/                    # LangGraph state machine
│   ├── graph.py             # Graph assembly, routing, checkpointer
│   ├── nodes.py             # Individual node implementations
│   ├── tools.py             # Retrieval (ChromaDB + BGE)
│   └── prompts.py           # System prompts
├── api/
│   ├── main.py              # FastAPI service
│   └── static/
│       └── index.html       # Frontend
├── ingestion/
│   ├── parse.py             # EPUB + PDF parsers, cleaner
│   ├── chunk.py             # Three chunking strategies (fixed, semantic, hierarchical)
│   └── embed_index.py       # BGE embedding + ChromaDB indexing
├── evaluation/
│   ├── ragas_eval.py        # Ragas evaluation harness
│   ├── gold_set.json        # Test questions with reference answers
│   └── reports/             # HTML reports (generated)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Scope note

Florence was originally scoped at ten primary works plus two critical companions (Virgil, Milton, Dostoevsky, Tolstoy, Bulgakov, Hemingway, Steinbeck, Morrison, Orwell, Huxley, Nabokov, Forster). The corpus was cut to two books (*1984* and *Brave New World*) so downstream work (evaluation, chunking analysis, grounding checks) could be done rigorously rather than gestured at.

Expanding the corpus is straightforward: drop the books into `Books/`, re-run `ingestion/embed_index.py`, and re-run the evaluation harness. The architecture does not need to change.

---

## Key findings

1. **Embedding model is the silent kingmaker.** Swapping `all-MiniLM-L6-v2` for `BAAI/bge-large-en-v1.5` moved retrieval quality from useless to reliable on the same corpus with the same downstream code. Everything downstream is downstream of what the retriever surfaces.

2. **Literary asymmetry in what RAG can do.** Orwell answers well because he narrates his themes plainly, meaning the words on the surface of the query match the words in the retrieved passage. Huxley answers less well because his themes live in irony and imagery, and dense retrieval matches on semantic surface rather than interpreted meaning. This is not a bug that can be eliminated; it is a property of the method.

3. **Right-sizing models matters.** Using Opus for every step is expensive and often wrong. Domain routing and chunk grading are yes/no decisions; Haiku handles them at a fraction of the cost. Generation and auditing warrant Sonnet. Choosing the right size for each step is the difference between a demo that costs cents and one that costs dollars per query.

---

## Known limitations

- **In-memory HITL checkpointer.** `MemorySaver` stores paused conversations in the container's RAM. This is fine for a single-container demo but would not survive a container restart or horizontal scaling. Correct upgrade path is a durable checkpointer backed by DynamoDB or Redis.
- **Hand-set thresholds.** The confidence threshold (0.7) and retry caps (2 rewrites, 3 rejections) are hand-set based on what worked during development. Production use would call for tuning against a larger evaluation set.
- **No streaming.** Generation waits for the full response before returning. Streaming would improve perceived latency for long answers.
- **Evaluation numbers are from broader-scope development.** Ragas scores were produced when the corpus included ten books; not yet re-run at the two-book scope.
---

## Tech stack

- **Language:** Python 3.12
- **Agent framework:** LangGraph (with `MemorySaver` for HITL persistence)
- **LLM provider:** Anthropic (Sonnet for generation and audit, Haiku for classification and grading, Opus as the offline evaluation judge)
- **Embeddings:** `BAAI/bge-large-en-v1.5` via `sentence-transformers` + `langchain-huggingface`
- **Vector store:** ChromaDB (local, persistent, HNSW index)
- **Web framework:** FastAPI + Pydantic + Uvicorn
- **Evaluation:** Ragas (reference-free variants)
- **Deployment:** Docker + AWS ECR + ECS Fargate + ALB + CloudWatch
- **Frontend:** Plain HTML, CSS, vanilla JavaScript (no build pipeline)


---

## License

MIT.

---

## Author

Built by Yohann Ian ([github.com/Yohann-Ian](https://github.com/Yohann-Ian)).

Portfolio write-up and design notes: ([techronin.art](https://techronin.art)).