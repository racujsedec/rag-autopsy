# Architecture Notes

## Design Goal
RAG Autopsy separates the RAG lifecycle into independently measurable stages:
1. Ingestion
2. Chunking
3. Indexing
4. Retrieval
5. Ranking / Reranking
6. Generation
7. Evaluation
8. Failure Diagnosis

This separation matters because an incorrect final answer does not necessarily mean the LLM failed.

## Phase 1 Baseline
Phase 1 implements document storage and deterministic fixed-size chunking so later methods have a reproducible baseline.

## Planned Failure Taxonomy
- ingestion failure
- chunk-boundary failure
- retrieval miss
- retrieval-depth failure
- ranking failure
- insufficient evidence
- generation hallucination
- citation mismatch
- ambiguous query
