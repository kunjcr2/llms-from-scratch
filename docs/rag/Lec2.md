# RAG Lec-1 Vizuara AI

### **Embedding strategies**

- **Recap**: Goal = grounded nutrition RAG; yesterday = parsing + 5 chunking types (fixed, semantic, structural, recursive, LLM).
- **Chunking we used**: **Fixed, 10 sentences per chunk** per page; optional overlap; drop tiny bits **(<30 tokens)** (footnotes etc.); result list = `pages_and_chunks_min_tokens`. Avg tokens stay under embedder limit.
- **Embed model**: `all-mpnet-base-v2` → **input >384 wordpieces gets truncated**, **output = 768-D**; 1 vector per chunk; save a CSV with `sentences` + `embedding`. Cosine sim = “meaning closeness.”
- **Speed**: CPU = minutes; **GPU ~10× faster**; batched GPU (e.g., 32) often a bit better—depends on HW; use GPU for 1k+ chunks.
- **Choosing embeddings**: check **max input**, **dim size** (bigger = richer but costly), **model size/latency**, **open/closed + cost/privacy**, and **domain** (multilingual, code/numerics, multimodal).
- **Where to store**:

  - Small (<~100k vecs): **numpy/torch + FAISS** (library, not a DB).
  - Managed vector DBs: **Pinecone/Qdrant/Weaviate/Milvus/Chroma** (easy, scalable; but it’s a second datastore).
  - **Postgres + pgvector**: one DB for **app data + vectors**, use **SQL** + vector indexes (IVF/HNSW). Great if you’re already on Postgres/Supabase.

- **Indexing 101**: **IVF/quantization** (search nearest clusters) vs **HNSW** (multi-layer graph). HNSW: faster queries/better recall, heavier to build; IVF: lighter builds, decent performance.
- **Runtime flow**: embed **query on the fly** → ANN search (cosine/dot/L2) → **top-k chunks** → augment LLM prompt.
- **Tips**: keep chunks under input limit, add small overlap, filter noise, batch on GPU, cache hot queries, try **hybrid (BM25+dense)** + **re-rankers**, always **eval end-to-end**.
- **Next**: hook up retrieval→generation, write **pgvector SQL** similarity, run evals, ship a **prod-ready** RAG.

## (2) Retrieval (R)

- Query → **embed with same model** as chunks → **similarity search**. Use `sentence_transformers.util.dot_score(query_emb, all_chunk_embs)` (fast on a plain `torch.tensor`; ~0.0x s for ~1.6k vecs). Grab **top-k** by score, return **(score, page_num, wrapped text)**. Dot product uses magnitude; `cos_sim` = angle-only (normed) — for text, both are common, I default to cosine in prod. I packaged it as `retrieve_relevant_sources(query, embeddings_matrix, model, k)` which encodes query, scores all chunks, and returns the top matches; sanity checks showed correct hits (e.g., macronutrients on p.5), with occasional confusions (macro vs micro) = **small embedder limits** → consider better embeddings or re-ranking.

## (3) Generation (G)

- Two paths: **open-source local** vs **closed-source API**. I demo’d **local** with HF’s **Gamma** (auto-select 2B-IT if ~15GB VRAM; 7B if you’ve got more), **quantized via bitsandbytes** to fit. Add HF **write token**, download, and test plain QA **without retrieval** to ensure the LLM runs. Trade-offs: local = privacy/control, smaller models; API = SOTA quality but $$ and external calls.

## (4) RAG = Retrieval-Augmented Generation (tie it all together)

- Pipeline: **query → retrieve top-k chunks → prompt_formatter(base_prompt, context_items, query) → LLM.generate → answer**. The **prompt formatter** appends the retrieved **context** to your base instruction (tell the model to only use provided context; if insufficient, say so). Worked well for “digest/absorption” style questions; when context didn’t directly answer (e.g., saliva case), model correctly said **“not enough from context”** → good sign (low hallucination).
- Practical notes: **chunking choice matters** (structural chunks were too big → blew the small LLM **context window**; semantic chunks = many tiny chunks → good relevance but smaller context passed). Keep **top-k small**, add **token budget** for context, consider **BM25+dense hybrid** and/or a **re-ranker** for higher precision.

## (5) Evaluation

- Build a small ground-truth sheet (20–50+ Qs with correct passages + answers). Use RAGAS to score: context precision/recall, entities recall, answer relevancy, faithfulness; add specificity/coverage if you like. For subjective bits, use LLM-as-judge but keep a human-in-the-loop to verify. Then sweep configs (chunking + overlap, k, embed model, index, re-ranker, prompt), run end-to-end, and pick by the avg scores; automate the grid + a tiny report. Re-run periodically to catch drift as your KB changes.
