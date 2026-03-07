# RAG Evaluation Maintenance Notes

The RAG evaluation surface should measure retrieval quality and answer behavior separately.

- Recall@K and MRR cover whether the expected source is retrieved.
- Keyword coverage checks whether the answer uses required facts.
- Scenario isolation verifies that cross-domain questions do not pollute context.
- Prompt profile accuracy confirms that high-risk questions use the intended guardrails.

A release should not rely on a single chat transcript. The regression datasets are the main evidence surface.
