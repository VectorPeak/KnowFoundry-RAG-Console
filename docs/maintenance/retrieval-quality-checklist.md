# Retrieval Quality Checklist

Use this checklist before activating a new knowledge-base version.

- Confirm FAQ direct answers still return exact policy wording for standard questions.
- Inspect source inference for short and ambiguous queries.
- Compare dense-only and hybrid retrieval examples when BM25 behavior changes.
- Review reranker output for low-score context pollution.
- Verify citations are present for document-backed answers.

The checklist is designed to catch regressions before they reach the console UI.
