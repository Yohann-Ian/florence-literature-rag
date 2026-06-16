# Florence - Literature RAG Agent

Florence is a production-style **agentic Retrieval-Augmented Generation (RAG)** system that answers interpretive questions about a curated corpus of comparative literature. It is built to be *grounded* - it answers from the texts it has, refuses questions outside its domain, and measures its own faithfulness rather than trusting that retrieval worked.

The project is as much about **evaluation and honest failure analysis** as it is about the agent itself. A working RAG demo is easy; knowing *how well* it works, *where* it breaks, and *why* is the harder and more interesting problem.

---

## What it does

- Accepts any natural-language question, but only **answers from its corpus** - out-of-domain questions are rejected by a domain router.
- Runs an **agentic retrieval loop**: retrieve → grade relevance → rewrite-and-retry if results are weak → generate → self-check for hallucination → escalate to human review when confidence is low.
- Exposes a **FastAPI** service with a human-in-the-loop (HITL) pause/resume flow that works over stateless HTTP.
- Ships as a **Docker** image with the embedding model and vector index baked in, ready to deploy.

---

## Architecture

```
Query
  │
  ▼
domain_router ──(out of domain)──► reject
  │ (valid)
  ▼
retrieve ◄──────────────┐
  │                     │
  ▼                     │
grade_documents ──(too few relevant)──► rewrite_query ──┘  (max 2 retries)
  │ (enough)
  ▼
generate
  │
  ▼
check_hallucination ──(ungrounded)──► regenerate (once)
  │ (grounded)
  ▼
hitl_checkpoint ──(low confidence)──► pause for human review
  │ (confident)
  ▼
Answer
```

The agent is built as a **LangGraph state machine** - a cyclic graph rather than a linear pipeline, which is what makes the retry loops and the human-in-the-loop pause possible.

### Stack

| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Anthropic Claude |
| Embeddings | BAAI/bge-large-en-v1.5 (local, CPU at serve time) |
| Vector store | ChromaDB |
| Evaluation | Ragas (faithfulness, answer relevancy, context precision) |
| Serving | FastAPI + Uvicorn |
| Packaging | Docker |

---

## The corpus

Twelve works, chosen around a single question - *what does civilisation do to the human soul?*

**Primary texts:** The Aeneid · Paradise Lost · Crime and Punishment · Anna Karenina · The Master and Margarita · A Farewell to Arms · Nineteen Eighty-Four · Brave New World · East of Eden · Beloved

**Critical texts:** Nabokov, *Lectures on Russian Literature* · Forster, *Aspects of the Novel*

The corpus is **closed by design**. Florence answers about *these* texts, deeply, rather than ingesting arbitrary uploads. This is a deliberate trade-off: a fixed corpus is what makes rigorous, repeatable evaluation possible. Live user ingestion would sacrifice the stable evaluation set that the whole project is built around.

---

## Evaluation & findings

Florence is evaluated against a hand-built gold set of 20 questions (17 answerable across single-text, cross-text-thematic, and cross-text-craft tiers; 3 adversarial). Scoring uses three **reference-free** Ragas metrics so results don't depend on model-drafted reference answers.

A few findings the evaluation surfaced - included here because the failures are more instructive than the successes:

**1. Retrieval quality is the dominant lever.** Swapping the embedding model from `all-MiniLM-L6-v2` (384-dim) to `BGE-large` (1024-dim) moved context precision on key questions from `0.00` to `1.00`, with fewer query rewrites needed. The bottleneck was never generation - it was retrieval pulling the wrong passages.

**2. A famous corpus is a hard case for grounding.** Because the underlying LLM knows these canonical texts well, it tends to answer from parametric knowledge rather than from retrieved chunks - producing answers that are *correct about the book* but *unfaithful to the sources*. The hallucination-check node exists to catch exactly this, and it does: low-faithfulness answers are flagged and escalated rather than shipped.

**3. Embeddings reward explicitness - which disadvantages some literature.** Orwell states his themes in plain expository prose, so passages about "control" retrieve reliably. Huxley conveys the same themes through irony and sensory imagery - meaning that lives in interpretation, not in the words - and retrieves poorly for the same query. Dense retrieval matches on semantic surface, not interpreted meaning. In a literary corpus this is not a bug to be fully eliminated but a property of the method that shapes which texts it can serve well.

**4. LLM-as-judge is usable but noisy.** Ragas and the agent's own hallucination check both use an LLM as judge. Running the same answers twice at temperature 0 produced stable aggregates (±0.05 per question), validating the method for coarse triage - while the two judges *disagreeing* on absolute faithfulness scores is itself a reminder not to over-trust any single number.

---

## Running locally

```bash
# Build the image
docker build -t florence-rag .

# Run the service (provide your Anthropic key via .env)
docker run --rm -p 8000:8000 --env-file .env florence-rag
```

Then open `http://127.0.0.1:8000/docs` for the interactive API.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/query` | Ask a question; returns an answer, a rejection, or a paused state |
| `POST` | `/resume` | Approve or correct a paused low-confidence answer |

---

## Known limitations & next steps

- **Chunking strategy is not yet empirically settled.** Hierarchical chunking is the working default; a three-way comparison (fixed / semantic / hierarchical) against the gold set is outstanding. The deeper issue is that all fixed-aperture chunking imposes a single notion of "the unit of meaning," whereas in literature that unit is genuinely variable - pointing toward comprehension-driven chunking as the real fix.
- **HITL state is in-memory.** Pause/resume works within a running server; a durable checkpointer (e.g. DynamoDB/Postgres) is needed for state to survive restarts.
- **Guardrails are a denylist.** The injection filter is a cheap first layer, not a real defence; the genuine protections are architectural (domain scoping, grounding, no world-acting tools).
- **No frontend yet.** The current interface is the FastAPI docs page; a proper UI is planned.

---

## Why this project

It sits at the intersection of two things I care about: creating **AI/ML systems** (evaluation, observability, deployment - not just a notebook) and the greatest works of world **literature** . Literature as a domain genuinely stresses the limits of current retrieval methods because meaning has many forms. One cannot simply mechanize it. It's worth looking into. The hard problems here: grounding against a corpus the model already knows, retrieving meaning that lives in subtext, are exactly the ones a generic technical-docs RAG never has to confront.
