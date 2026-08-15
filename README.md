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

## Current Capabilities

RAG Autopsy currently includes:

- fixed-size and paragraph-aware chunking
- BM25, semantic, hybrid RRF, and cross-encoder reranking
- evidence-based ground truth
- answerable and unanswerable benchmark questions
- Recall@1, Recall@3, and MRR evaluation
- retrieval, ranking, chunk-boundary, and context-loss diagnosis
- previous-chunk context enrichment
- PostgreSQL + pgvector persistent vector retrieval
- separate canonical and retrieval-context storage
- reproducible pgvector schema, indexing, and parity benchmark
- automated unit tests

The current focus is retrieval quality and failure diagnosis before LLM generation.

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

## Benchmark v1

Benchmark v1 contains **24 documents and 42 questions**: 36 answerable and 6 unanswerable.

### Chunking comparison

| Strategy | Recall@1 | Recall@3 | MRR | Evidence coverage |
|---|---:|---:|---:|---:|
| Fixed-size | 66.7% | 88.9% | 0.750 | 94.9% |
| Paragraph-aware | **80.6%** | **91.7%** | **0.861** | **99.3%** |

The autopsy engine also identified **CHUNK_CONTEXT_LOSS**: cases where the answer evidence is completely preserved, but identifying query context is separated into an adjacent chunk.

### Context enrichment

`PreviousChunkContextEnricher` adds the previous chunk as retrieval context while preserving the current chunk ID and boundaries.

| Configuration | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| Semantic original | 83.3% | 88.9% | 0.861 |
| **Semantic + previous-chunk context** | **83.3%** | **100.0%** | **0.917** |

Four previous semantic retrieval misses moved into the top 3 with no rank regressions in this comparison.

Reproduce it with:

`python scripts/compare_context_enrichment.py`

## PostgreSQL + pgvector

RAG Autopsy supports persistent semantic retrieval with PostgreSQL and pgvector.

Canonical chunk text is stored separately from the context-enriched text used to create embeddings:

```text
canonical paragraph        -> text
previous + current         -> retrieval_text
retrieval_text             -> vector(384) embedding
```

This preserves clean chunk text for future citations and generation while allowing richer retrieval context.

### Database setup

```bash
pip install -e ".[dev,postgres]"
createdb rag_autopsy
psql -d rag_autopsy -f sql/pgvector_schema.sql
python scripts/index_pgvector.py
```

A different database connection can be supplied with `RAG_AUTOPSY_DATABASE_URL`.

### Contextual pgvector parity

| Retriever | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| In-memory contextual | 83.3% | 100.0% | 0.917 |
| PostgreSQL pgvector | 83.3% | 100.0% | 0.917 |

Across all 36 answerable Benchmark v1 questions, the complete ordered top-3 rankings produced **0 mismatches** between contextual in-memory semantic retrieval and contextual PostgreSQL pgvector retrieval.

Reproduce the comparison with:

```bash
python scripts/compare_pgvector.py
```

## Roadmap

### Phase 1 — Foundations
- [x] repository structure
- [x] benchmark corpus
- [x] fixed-size chunking
- [x] paragraph-aware chunking
- [x] automated tests

### Phase 2 — Retrieval Baselines
- [x] BM25 retrieval
- [x] semantic embedding retrieval
- [x] retrieval evaluation dataset
- [x] Recall@K and MRR
- [x] hybrid RRF retrieval
- [x] cross-encoder reranking

### Phase 3 — Retrieval Diagnostics
- [x] retrieval-miss diagnosis
- [x] ranking-failure diagnosis
- [x] reranker improvement/regression diagnosis
- [x] chunk-boundary diagnosis
- [x] chunk context-loss diagnosis
- [x] previous-chunk context enrichment
- [x] PostgreSQL + pgvector persistent retrieval
- [x] in-memory / pgvector parity validation

### Phase 4 — Grounded Generation
- [x] LLM abstraction layer
- [x] OpenAI Responses API adapter
- [x] grounded answer generation
- [x] chunk-ID citations
- [x] citation validity diagnosis
- [x] textual citation-support diagnosis
- [x] claim-level citation coverage

### Phase 5 — RAG Autopsy Engine
- [x] end-to-end autopsy report
- [x] stage-aware primary verdict
- [x] human-readable report formatter
- [x] CLI workflow
- [x] answerable and unanswerable benchmark support

### Phase 6 — Productization
- [x] FastAPI service
- [x] typed OpenAPI request/response schemas
- [x] API validation and service-error handling
- [x] benchmark question API
- [x] React + TypeScript diagnostic UI
- [x] configurable Top K
- [x] retrieval-only browser workflow
- [x] grounded-generation browser workflow
- [ ] Docker packaging
- [ ] CI/CD
- [ ] hosted deployment

### Possible future experiments

These are intentionally outside the v1 completion scope:

- nDCG and additional ranking metrics
- semantic or section-aware chunking
- experiment tracking
- Docker deployment
- GitHub Actions
- hosted cloud deployment

## Engineering Principle

Every experiment should answer a measurable question.

Example:

> Does semantic chunking improve Recall@5 over fixed-size chunking on our evaluation set?

That principle keeps the project focused on engineering evidence rather than framework demos.
