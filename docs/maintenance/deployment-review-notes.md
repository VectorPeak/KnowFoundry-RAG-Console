# Deployment Review Notes

Deployment review should focus on dependencies that affect startup and retrieval stability.

- Milvus, MySQL, local embedding model, reranker model, and LLM credentials are required.
- The active knowledge-base version must exist before the API is treated as ready.
- Docker Compose and local development mode use different host and path conventions.
- Environment examples should keep placeholders only and never include real keys.

These notes keep deployment checks explicit rather than hidden in one-off shell history.
