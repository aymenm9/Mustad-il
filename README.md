# Mustad-il - Islamic Text Search Engine

A clean, efficient, and semantic search engine for the Quran and Hadith, powered by simple yet effective IR algorithms and optional LLM query generation.

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    > **Note:** Requires a valid `GEMINI_API_KEY` in a `.env` file for the AI query generation features.

2.  **Start the Application:**
    Navigate to the `src` directory and run:
    ```bash
    cd src
    python main.py
    ```

3.  **Open in Browser:**
    Open `http://localhost:8000` or simply drag and drop `index.html` into your browser.

## 💡 How It Works

This search engine primarily uses **BM25 (Best Matching 25)**, which is arguably the best "traditional" ranking function for information retrieval.

*   **Why BM25?** We initially tested **TF-IDF**, but the results were inconsistent for the nuanced language of religious texts. BM25 generally provides much better relevance ranking by handling term saturation more effectively.
*   **Library vs. Custom:** You will see files ending in `_lib.py`. These use optimized libraries (like `rank_bm25`). In our testing, the results between our custom implementations and the libraries were nearly identical, the samme go for from algorithm to algorithm spetily with bigger queries.

## 📂 Project Structure

```
├── index.html               # Frontend interface
├── requirements.txt         # Project dependencies
└── src/
    ├── main.py              # Main entry point (FastAPI app)
    ├── gemini_llm.py        # AI logic for generating search queries
    ├── run_user_query.py    # Search execution logic
    ├── load_engines.py      # Module to load indexes
    ├── schemas.py           # Data models (Pydantic)
    ├── preprocessing.py     # Arabic text cleaner/processor
    ├── indexing.py          # Builds search indexes from raw JSON
    ├── bm25_search.py       # Custom BM25 implementation
    ├── bm25_search_lib.py   # Library-based BM25 implementation
    ├── tfidf_search.py      # Custom TF-IDF implementation
    ├── tfidf_search_lib.py  # Library-based TF-IDF implementation
    ├── vsm_search.py        # Vector Space Model implementation
    ├── vsm_search_lib.py    # Library-based VSM implementation
    └── indices/             # Generated index files (auto-created)
```
