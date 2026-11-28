# Semantic Retriever: From Keywords to Transformers

The retrieval backbone of an internal RAG (Retrieval-Augmented Generation) assistant. Given an employee question, it finds the most relevant company documents so a language model can answer from them. Three retrievers are built and compared on the same data:

| Era | Method | Idea |
|---|---|---|
| Keywords | **BM25** | Score documents by exact term overlap (TF, IDF, length normalization) |
| Static embeddings | **Word2Vec** | Average one fixed vector per word into a document vector |
| Contextual embeddings | **Transformer** (`sentence-transformers`) | Encode whole sentences with self-attention, compare by cosine similarity |

Full analysis: [`unified_retrieval_comparison.ipynb`](unified_retrieval_comparison.ipynb).
Interactive demo: [`streamlit/app.py`](streamlit/app.py).

---

## 1. The retriever's role in a RAG system

A RAG system answers questions in **two stages: retrieval, then generation**.

```
 question ──► RETRIEVER ──► top-k relevant documents ──► GENERATOR (LLM) ──► grounded answer
              (this project)   "what to read"              "how to say it"
```

1. **Retrieval:** the retriever searches the document collection and selects the few documents most relevant to the question.
2. **Generation:** those documents are placed in the language model's prompt as context, and the model writes an answer based on them.

The retriever is the gatekeeper for the generator. The model can only use what it is handed, so:

- If the right document is **not retrieved**, the generator cannot answer correctly, however strong it is. It either says it doesn't know or guesses (hallucinates).
- If **irrelevant** documents are retrieved, they add noise and can pull the answer off course.

The notebook shows this on a real query, *"who played bailey in the sisterhood of the traveling pants"*. BM25 ranked an article about a different actor (Jesse Williams) first. It matches the rare words "sisterhood", "traveling" and "pants", but never contains "bailey" or "played", so it cannot answer the question, and BM25 scored it slightly higher than the correct passage (34.1 vs 32.1). Word2Vec and the transformer ranked the correct passage, the one about Jenna Boyd, first. A generator given BM25's top result would have had no correct answer to work from.

**Real-world examples.** The same retrieve-then-generate pattern underlies assistants that answer from external sources, such as **Perplexity** (searches the web and answers with cited sources), **Microsoft Copilot** (grounds answers in web search results and, in Microsoft 365, in a user's own documents) and **ChatGPT** (its search and file-search features pull in relevant content before answering). For these products, as for this project, answer quality depends heavily on retrieval quality.

---

## 2. From keyword search (BM25) to semantic search (transformers)

**BM25 (keywords matching).** The standard text search is fast, interpretable and needs no training, but it only matches the *words* in the query. It cannot tell "who plays X" and "the actor cast as X" mean the same thing, and it is distracted by documents that share common words but not the meaning.

**Word2Vec (static embeddings).** A step toward meaning: words used in similar contexts get similar vectors, so related words end up close together. But each word has **one fixed vector**, so "bank" (money) and "bank" (river) are identical, and the model must be trained on the corpus itself. Here the corpus is small, so quality depends heavily on tuning.

**Transformers (contextual embeddings).** Self-attention lets the model read the whole sentence at once, so a word's representation depends on its context. Models pre-trained on massive text already "know" language, so they need no training on your corpus. Queries and documents are mapped into the same vector space and ranked by cosine similarity, so a question and a passage can match even when they share few words. The costs are heavier compute (see the speed table below) and less transparent scoring than BM25.

This is why the field moved from keyword matching → static embeddings → pre-trained contextual embeddings.

---

## 3. Performance comparison

**Setup (from the notebook):** Natural Questions (BEIR), 2,000 randomly sampled test queries (seed 42) and the 2,440 documents that are relevant to them. Metrics are computed by `src/evaluator.py`; k is the cutoff for the top-k results.

| Method | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.5328 | 0.7266 | 0.7740 | 0.1751 | 0.6738 |
| Word2Vec (tuned) | 0.5503 | 0.7658 | 0.8142 | 0.1877 | 0.6999 |
| Transformer (MiniLM-L6) | 0.7760 | 0.9298 | 0.9497 | 0.2263 | 0.9044 |

**Findings**

- **Transformers win on every metric.** MiniLM improves MRR over BM25 by 34% (0.674 → 0.904) and finds the right document first (Recall@1) 78% of the time versus 53%.
- **Word2Vec only edges out BM25 once tuned.** With default settings it was nearly useless (MRR 0.0255, Recall@5 0.030). A random search over 10 configurations found a much better
  one: skip-gram (`sg=1`), `vector_size=100`, `window=20`, `min_count=2`, `epochs=100`, giving MRR 0.6999. Even then it is only about 4% better than BM25 on MRR, because a ~2,400-document corpus is too small to learn good word vectors.
- **Precision@5 looks low for every method by design.** Most queries have about one relevant document, so at most 1 of the top 5 can be relevant (the ceiling is roughly 0.2). Read Precision@5 relative to that ceiling. Recall and MRR are the more informative metrics here.

### Which Retrieval Method?

| Model | Recall@5 | Recall@10 | MRR | Index build | Retrieval (2,000 queries) |
|---|---|---|---|---|---|
| MiniLM-L6-v2 (~23M params, 384-d) | 0.9298 | 0.9497 | 0.9044 | 41.8 s | 2.9 s |
| BGE-small-en-v1.5 (~33M, 384-d) | 0.9382 | 0.9541 | 0.9067 | 99.0 s | 5.8 s |
| **MPNet-base-v2 (~109M, 768-d)** | **0.9587** | **0.9705** | **0.9297** | 295.3 s | 19.1 s |

MPNet is the most accurate (+38% MRR over BM25) but about 7x slower to index and 6.6x slower to query than MiniLM. MiniLM is the best speed/accuracy trade-off. Timings were measured, but this could very per machine capacity. Only MPNet reaches the project's 95–99% recall target (Recall@5 95.9%, Recall@10 97.1%); MiniLM comes close (Recall@10 95.0%).

### Caveats

- **The corpus is easier than the real task.** It contains only documents relevant to the sampled queries (2,440 documents), not the full ~2.7M-passage NQ corpus, so absolute scores would be lower in a realistic setting. The *ranking* of the methods is the reliable result.
- **Word2Vec was tuned on the evaluation queries**, so its score is somewhat optimistic. The transformers were used off the shelf without tuning.
- Results are for one sample of 2,000 queries with one seed.

---

## 4. Project layout

```
.
├── unified_retrieval_comparison.ipynb   # Full comparison and analysis
├── src/
│   ├── data_loader.py                   # BEIR / Natural Questions loader (Hugging Face)
│   ├── bm25_retriever.py                # BM25 keyword retriever
│   ├── word2vec_retriever.py            # Word2Vec retriever + hyperparameter search
│   ├── transformer_retriever.py         # Sentence-transformer retriever
│   ├── evaluator.py                     # Recall@k, Precision@k, MRR
│   └── utils.py                         # Comparison tables and plots
├── streamlit/                           # Interactive demo (app.py, demo_data.py, rag_system.py)
├── tests/                               # Unit tests for each retriever and the metrics
├── utils/                               # Logging
├── docs/                                # mission.md, roadmap.md
└── tasks/                               # Task descriptions per phase
```

## 5. Run it

```bash
pip install -r requirements.txt

# Tests (20 tests)
python -m pytest tests/ -v

# Notebook (downloads Natural Questions from Hugging Face on first run)
jupyter lab unified_retrieval_comparison.ipynb

# Demo: run from the project root
streamlit run streamlit/app.py
```

**Demo.** The Streamlit app runs all three retrievers side by side on a small demo corpus (a fictional company, "TechFlow AI"), showing each method's top documents, whether they are correct, and precision/recall for the query. It uses the MPNet transformer. Optionally, set `OPENAI_API_KEY` (see `streamlit/.env.example`) to also generate an answer from each retriever's documents, which makes the retrieval-then-generation link concrete. The first launch downloads the MPNet model.

## 6. Scope

Built for internal use only: employees (summaries and clarifications), managers (quick inventory lookups) and IT (search for troubleshooting guides). It is not intended for external or commercial use and does not replace professional expertise. See [`docs/mission.md`](docs/mission.md).
