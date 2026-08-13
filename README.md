# RAG Autopsy

**RAG Autopsy** is a retrieval diagnostics and evaluation platform for investigating *why* Retrieval-Augmented Generation systems fail.

Instead of only asking whether a RAG system produced a good answer, RAG Autopsy is designed to identify the failure stage:

- document ingestion
- chunking
- retrieval
- ranking
- evidence coverage
- generation
- citation grounding

## Why this project exists

Many RAG demos stop at:

> retrieve documents → send context to an LLM → show an answer

Production systems need a deeper question:

> When the answer is wrong, what failed?

RAG Autopsy will provide controlled experiments, retrieval metrics, failure classification, and eventually an interactive diagnostic UI.

## Planned Architecture

```text
Documents
   |
   v
Ingestion
   |
   v
Chunking
   |
   v
Embeddings + Lexical Index
   |
   v
Vector / BM25 / Hybrid Retrieval
   |
   v
Reranking
   |
   v
LLM Generation
   |
   v
Evaluation + Failure Diagnosis
```

## Phase 1

Current milestone:

- repository structure
- Python package
- small evaluation corpus
- baseline fixed-size chunker
- unit tests
- no LLM dependency

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Roadmap

### Phase 1 — Foundations
- [x] repository structure
- [x] baseline corpus
- [x] fixed-size chunking
- [x] basic tests

### Phase 2 — Retrieval Baselines
- [ ] lexical/BM25 retrieval
- [ ] embedding retrieval
- [ ] retrieval evaluation dataset
- [ ] Recall@K / MRR / nDCG

### Phase 3 — Retrieval Quality
- [ ] semantic chunking
- [ ] section-aware chunking
- [ ] hybrid search
- [ ] reranking
- [ ] experiment tracking

### Phase 4 — Generation
- [ ] LLM abstraction layer
- [ ] grounded answer generation
- [ ] citations
- [ ] answer-level evaluation

### Phase 5 — Autopsy Engine
- [ ] failure taxonomy
- [ ] automatic failure classification
- [ ] chunking failure detection
- [ ] retrieval failure detection
- [ ] ranking failure detection
- [ ] hallucination / citation mismatch detection

### Phase 6 — Productization
- [ ] FastAPI
- [ ] React UI
- [ ] Docker
- [ ] GitHub Actions
- [ ] AWS deployment

## Engineering Principle

Every experiment should answer a measurable question.

Example:

> Does semantic chunking improve Recall@5 over fixed-size chunking on our evaluation set?

That principle keeps the project focused on engineering evidence rather than framework demos.
