# Module Structure and Dependencies

## File Organization

```
harage_projet_riw/
│
├── preprocessing.py          # Core preprocessing (no dependencies)
│   └── SafeIslamicArabicProcessor
│
├── indexing.py              # Index building (depends on: preprocessing)
│   ├── build_quran_index()
│   ├── build_hadith_index()
│   ├── build_inverted_index_quran()
│   ├── build_inverted_index_hadith()
│   ├── save_index()
│   └── save_inverted_index()
│
├── tfidf_search.py          # TF-IDF search (depends on: preprocessing)
│   └── TFIDFSearchEngine
│
├── bm25_search.py           # BM25 search (depends on: preprocessing)
│   └── BM25SearchEngine
│
├── vsm_search.py            # VSM search (depends on: preprocessing)
│   └── VectorSpaceModel
│
├── gemini_llm.py            # LLM integration (depends on: bm25_search)
│   ├── generate_strict_queries()
│   └── search_with_queries()
│
├── main.py                  # Orchestration (depends on: all above)
│   └── main()
│
├── examples.py              # Usage examples (depends on: main, gemini_llm)
│   ├── example_basic_search()
│   ├── example_llm_search()
│   └── example_compare_engines()
│
└── README.md                # Documentation
```

## Dependency Graph

```
                    preprocessing.py
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    indexing.py    tfidf_search.py   bm25_search.py   vsm_search.py
          │                │                │                │
          └────────────────┴────────────────┴────────────────┘
                                   │
                                   ▼
                            gemini_llm.py
                                   │
                                   ▼
                               main.py
                                   │
                                   ▼
                             examples.py
```

## Module Responsibilities

### Layer 1: Core (No Dependencies)
- **preprocessing.py**: Arabic text preprocessing, tokenization, normalization

### Layer 2: Data Processing (Depends on Layer 1)
- **indexing.py**: Build and save document indices and inverted indices
- **tfidf_search.py**: TF-IDF ranking algorithm
- **bm25_search.py**: BM25 ranking algorithm  
- **vsm_search.py**: Vector Space Model with cosine similarity

### Layer 3: Intelligence (Depends on Layer 2)
- **gemini_llm.py**: LLM-powered query generation and execution

### Layer 4: Orchestration (Depends on All)
- **main.py**: Initialize all components and return engines
- **examples.py**: Demonstrate usage patterns

## Import Patterns

### To use just preprocessing:
```python
from preprocessing import SafeIslamicArabicProcessor
```

### To use indexing:
```python
from preprocessing import SafeIslamicArabicProcessor
from indexing import build_quran_index, build_hadith_index
```

### To use a specific search engine:
```python
from preprocessing import SafeIslamicArabicProcessor
from bm25_search import BM25SearchEngine
```

### To use everything:
```python
from main import main
engines = main()
```

### To use LLM features:
```python
from main import main
from gemini_llm import generate_strict_queries, search_with_queries

engines = main()
queries = generate_strict_queries("your question")
results = search_with_queries(queries, engines['bm25_quran'], engines['bm25_hadith'])
```

## Key Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Minimal Dependencies**: Modules only import what they need
3. **Reusability**: Each module can be used independently
4. **Consistency**: All search engines use the same preprocessor
5. **Extensibility**: Easy to add new search algorithms or features

## Data Flow

```
Raw Text (Quran/Hadith JSON)
         ↓
    preprocessing.py (SafeIslamicArabicProcessor)
         ↓
    Tokenized Documents
         ↓
    indexing.py (build indices)
         ↓
    Document Index + Inverted Index
         ↓
    Search Engines (TF-IDF, BM25, VSM)
         ↓
    Ranked Results
         ↓
    gemini_llm.py (optional: query enhancement)
         ↓
    Final Results
```
