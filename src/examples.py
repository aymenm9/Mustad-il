from load_engines import load_engines_fast
from gemini_llm import GeminiSearchModel
import time


def example_basic_search():
    print("=" * 70)
    print("EXAMPLE 1: Basic BM25 Search (Custom vs Library)")
    print("=" * 70)
    
    engines = load_engines_fast()
    
    query = "اين الله"  
    print(f"\nSearching for: '{query}'\n")
    
    print("📖 QURAN - Custom BM25:")
    results = engines['bm25_quran'].search(query, top_k=2)
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        print(f"{i}. Surah {meta['chapter']}:{meta['verse']} (Score: {result['score']:.4f})")
        print(f"   {result['text']}\n")
    
    print("📖 QURAN - Library BM25:")
    results = engines['bm25_quran_lib'].search(query, top_k=2)
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        print(f"{i}. Surah {meta['chapter']}:{meta['verse']} (Score: {result['score']:.4f})")
        print(f"   {result['text']}\n")
    
    print("📚 HADITH - Custom BM25:")
    results = engines['bm25_hadith'].search(query, top_k=2)
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        print(f"{i}. {meta['book']} #{meta['hadith_number']} (Score: {result['score']:.4f})")
        print(f"   {result['text']}\n")
    
    print("📚 HADITH - Library BM25:")
    results = engines['bm25_hadith_lib'].search(query, top_k=2)
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        print(f"{i}. {meta['book']} #{meta['hadith_number']} (Score: {result['score']:.4f})")
        print(f"   {result['text']}\n")


def example_llm_search_all_engines():
    print("\n" + "=" * 70)
    print("EXAMPLE 2: LLM Query Generation + All Engines Comparison")
    print("=" * 70)
    
    engines = load_engines_fast()
    model = GeminiSearchModel() # Use the class directly
    
    question="من اخر نبی"
    print(f"\nUser Question: '{question}'\n")
    
    print("Step 1: Generating queries...")
    queries = model.generate_queries(question) # Call method on instance
    
    print(f"Generated {len(queries)} queries:")
    for q in queries:
        print(f"  - [{q['type']}] {q['query']}")
    
    print("\n" + "=" * 70)
    print("Step 2: Testing each query with ALL engines")
    print("=" * 70)
    
    for idx, q in enumerate(queries, 1):
        query_text = q['query']
        query_type = q['type']
        
        print(f"\n{'='*70}")
        print(f"Query {idx}: '{query_text}' (Type: {query_type.upper()})")
        print(f"{'='*70}")
        
        if query_type == "quran":
            engines_to_test = [
                ("Custom TF-IDF", engines['tfidf_quran']),
                ("Library TF-IDF", engines['tfidf_quran_lib']),
                ("Custom BM25", engines['bm25_quran']),
                ("Library BM25", engines['bm25_quran_lib']),
                ("Custom VSM", engines['vsm_quran']),
                ("Library VSM", engines['vsm_quran_lib']),
            ]
        else:
            engines_to_test = [
                ("Custom TF-IDF", engines['tfidf_hadith']),
                ("Library TF-IDF", engines['tfidf_hadith_lib']),
                ("Custom BM25", engines['bm25_hadith']),
                ("Library BM25", engines['bm25_hadith_lib']),
                ("Custom VSM", engines['vsm_hadith']),
                ("Library VSM", engines['vsm_hadith_lib']),
            ]
        
        for engine_name, engine in engines_to_test:
            print(f"\n🔍 {engine_name}:")
            
            start = time.time()
            results = engine.search(query_text, top_k=2)
            search_time = (time.time() - start) * 1000
            
            print(f"   Time: {search_time:.2f}ms")
            
            if results:
                for i, res in enumerate(results, 1):
                    if isinstance(res, tuple):
                        doc_id, score, doc = res
                        meta = doc
                    else:
                        score = res.get('score', 0)
                        meta = res.get('metadata', res)
                    
                    if query_type == "quran":
                        print(f"   {i}. Surah {meta.get('chapter')}:{meta.get('verse')} (Score: {score:.4f})")
                    else:
                        print(f"   {i}. {meta.get('book')} #{meta.get('hadith_number')} (Score: {score:.4f})")
                    
                    text = meta.get('arabic_original', '') or res.get('text', '') if not isinstance(res, tuple) else doc.get('arabic_original', '')
                    print(f"      {text}")
            else:
                print("   No results found")


def example_llm_validated_search():
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Validated Search with LLM (Two-Step Process)")
    print("=" * 70)
    
    engines = load_engines_fast()
    model = GeminiSearchModel()
    
    question = "اين الله"
    print(f"\nUser Question: '{question}'\n")
    
    print("Step 1: Generating queries...")
    queries = model.generate_queries(question)
    
    if not queries:
        print("No queries generated.")
        return
        
    q = queries[0]
    query_text = q['query']
    query_type = q['type']
    
    print(f"\nUsing Top Query: '{query_text}' (Type: {query_type.upper()})")
    
    print("\n🔍 Running IR Search (BM25)...")
    if query_type == "quran":
        results = engines['bm25_quran'].search(query_text, top_k=3)
    else:
        results = engines['bm25_hadith'].search(query_text, top_k=3)
        
    # Extract text content for validation
    texts = [res['text'] for res in results]
        
    print(f"Found {len(texts)} candidates. Running LLM Validation...")
    
    # Use the new filter_results method which returns only valid texts
    valid_texts_list = model.filter_results(question, query_text, texts)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS (Validated by LLM)")
    print("=" * 70)
    
    # We need to map back to original metadata to print nicely
    # This is a bit tricky since filter_results returns strings, but we can fuzzy match or just print what we have.
    # For a robust solution, filter_results logic might need to be different in real app, but for example:
    
    relevant_count = len(valid_texts_list)
    
    # Simple set for checking membership
    valid_texts_set = set(valid_texts_list)
    
    for i, res in enumerate(results, 1):
        text_content = res['text']
        is_rel = text_content in valid_texts_set
        
        status = "✅ RELEVANT" if is_rel else "❌ NOT RELEVANT"
        print(f"\n{i}. [{status}]")
        
        meta = res['metadata']
        if query_type == "quran":
            print(f"   Source: Surah {meta['chapter']}:{meta['verse']}")
        else:
            print(f"   Source: {meta['book']} #{meta['hadith_number']}")
            
        print(f"   Text: {res['text'][:150]}...")
        
    print(f"\nSummary: {relevant_count}/{len(results)} results validated as relevant.")


def example_compare_engines():
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Comparing Search Engines")
    print("=" * 70)
    
    engines = load_engines_fast()
    
    query = "الله الرحيم"
    print(f"\nQuery: '{query}'\n")
    
    print("🔍 Custom TF-IDF:")
    results = engines['tfidf_quran'].search(query, top_k=2)
    for i, (doc_id, score, doc) in enumerate(results, 1):
        print(f"{i}. Surah {doc['chapter']}:{doc['verse']} (Score: {score:.4f})")
        print(f"   {doc['arabic_original']}\n")
    
    print("🔍 Library TF-IDF:")
    results = engines['tfidf_quran_lib'].search(query, top_k=2)
    for i, (doc_id, score, doc) in enumerate(results, 1):
        print(f"{i}. Surah {doc['chapter']}:{doc['verse']} (Score: {score:.4f})")
        print(f"   {doc['arabic_original']}\n")
    
    print("🔍 Custom BM25:")
    results = engines['bm25_quran'].search(query, top_k=2)
    for i, res in enumerate(results, 1):
        meta = res['metadata']
        print(f"{i}. Surah {meta['chapter']}:{meta['verse']} (Score: {res['score']:.4f})")
        print(f"   {res['text']}\n")
    
    print("🔍 Library BM25:")
    results = engines['bm25_quran_lib'].search(query, top_k=2)
    for i, res in enumerate(results, 1):
        meta = res['metadata']
        print(f"{i}. Surah {meta['chapter']}:{meta['verse']} (Score: {res['score']:.4f})")
        print(f"   {res['text']}\n")


if __name__ == "__main__":
    #example_basic_search()
    example_llm_search_all_engines()
    #example_compare_engines()
    #example_llm_validated_search()
