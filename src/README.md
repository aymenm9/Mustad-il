# Islamic Text Search Engine - Modular Structure

This project provides a comprehensive search engine for Islamic texts (Quran and Hadith) with multiple search algorithms.

## Project Structure

```
.
├── preprocessing.py          # Arabic text preprocessing
├── indexing.py              # Index building and management
├── tfidf_search.py          # TF-IDF search engine (custom)
├── tfidf_search_lib.py      # TF-IDF search engine (scikit-learn)
├── bm25_search.py           # BM25 search engine (custom)
├── bm25_search_lib.py       # BM25 search engine (rank-bm25)
├── vsm_search.py            # Vector Space Model (custom)
├── vsm_search_lib.py        # Vector Space Model (scikit-learn)
├── gemini_llm.py            # Gemini API integration
├── main.py                  # Main orchestration script
├── examples.py              # Usage examples
├── comparison_examples.py   # Performance comparison demos
└── COMPARISON.md            # Detailed comparison guide
```

## Two Implementations Available

This project provides **both custom and library-based** implementations:

- **Custom** (`*_search.py`): Educational, transparent, full control
- **Library** (`*_search_lib.py`): Production-ready, optimized, faster

See [COMPARISON.md](COMPARISON.md) for detailed comparison.

## Module Descriptions

### 1. **preprocessing.py**
Contains the `SafeIslamicArabicProcessor` class for Arabic text preprocessing:
- Removes diacritics
- Normalizes Arabic text
- Protects sacred terms and phrases
- Tokenizes text safely

### 2. **indexing.py**
Handles all indexing operations:
- `build_quran_index()` - Build Quran document index
- `build_hadith_index()` - Build Hadith document index
- `build_inverted_index_quran()` - Create inverted index for Quran
- `build_inverted_index_hadith()` - Create inverted index for Hadith
- `save_index()` - Save indices to JSON and CSV
- `save_inverted_index()` - Save inverted indices to JSON

### 3. **tfidf_search.py**
Implements TF-IDF (Term Frequency-Inverse Document Frequency) search:
- Classic information retrieval algorithm
- Good for general text search

### 4. **bm25_search.py**
Implements BM25 (Best Matching 25) search:
- State-of-the-art probabilistic ranking function
- Better handling of term frequency saturation
- Recommended for most use cases

### 5. **vsm_search.py**
Implements Vector Space Model search:
- Uses cosine similarity
- TF-IDF weighted vectors
- Good for semantic similarity

### 6. **gemini_llm.py**
Integrates Google Gemini API for intelligent query generation:
- `generate_strict_queries()` - Convert natural language questions to search queries
- `search_with_queries()` - Execute searches using generated queries
- Automatically separates Quran and Hadith queries

### 7. **main.py**
Main orchestration script that:
- Initializes the preprocessor
- Builds all indices
- Creates all search engines
- Returns a dictionary of all components

## Usage Examples

### Basic Usage

```python
from main import main

# Initialize all components
engines = main()

# Access individual components
processor = engines['processor']
bm25_quran = engines['bm25_quran']
bm25_hadith = engines['bm25_hadith']

# Perform a search
results = bm25_quran.search("الصلاة", top_k=5)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text']}\n")
```

### Using Library-Based Implementations (Faster)

```python
from main import main

engines = main()

# Use library-based BM25 (rank-bm25) - faster!
results = engines['bm25_quran_lib'].search("الصلاة", top_k=5)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text']}\n")

# Compare performance
from comparison_examples import compare_search_engines
compare_search_engines()
```

### Using Individual Modules

```python
# Just preprocessing
from preprocessing import SafeIslamicArabicProcessor

processor = SafeIslamicArabicProcessor()
result = processor.preprocess("بسم الله الرحمن الرحيم")
print(result['tokens'])

# Just indexing
from indexing import build_quran_index
from preprocessing import SafeIslamicArabicProcessor

processor = SafeIslamicArabicProcessor()
quran_index = build_quran_index('qoran/quran.json', processor)

# Just BM25 search
from bm25_search import BM25SearchEngine
from preprocessing import SafeIslamicArabicProcessor

processor = SafeIslamicArabicProcessor()
# ... load your indices ...
bm25 = BM25SearchEngine("Quran", inverted_index, doc_metadata, processor)
results = bm25.search("الله", top_k=10)
```

### Using Gemini LLM for Query Generation

```python
from gemini_llm import generate_strict_queries, search_with_queries
from main import main

# Initialize engines
engines = main()

# Generate queries from natural language
question = "ما هي أركان الإسلام؟"
queries = generate_strict_queries(question)

# Execute searches
results = search_with_queries(
    queries, 
    engines['bm25_quran'], 
    engines['bm25_hadith'],
    top_k=5
)

# Display results
for result in results:
    print(f"Query: {result['query']}")
    print(f"Type: {result['type']}")
    for search_result in result['search_results']:
        print(f"  - {search_result['text']}")
```

## Dependencies

### Core Dependencies
```
pandas
google-genai
camel-tools
pyarabic
```

### Library-Based Implementations
```
scikit-learn>=1.3.0
rank-bm25>=0.2.2
numpy>=1.24.0
```

## Data Structure

### Input Data
- **Quran**: `qoran/quran.json`
- **Hadith**: `hadith/bukhari.json`, `hadith/muslim.json`, `hadith/malik.json`

### Output Indices
All indices are saved in the `indices/` directory:
- `quran_index.json` / `quran_index.csv`
- `hadith_index.json` / `hadith_index.csv`
- `quran_inverted_index.json`
- `hadith_inverted_index.json`

## Search Engine Comparison

| Engine | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| TF-IDF | General search | Fast | Good |
| BM25 | Most use cases | Fast | Excellent |
| VSM | Semantic similarity | Medium | Very Good |

## Notes

- All search engines use the same preprocessor for consistency
- BM25 is recommended for production use
- Gemini LLM integration requires an API key
- Sacred terms and phrases are protected during preprocessing
