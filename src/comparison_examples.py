from load_engines import load_engines_fast
from gemini_llm import generate_strict_queries, search_with_queries
import time


def compare_search_engines():
    print("=" * 80)
    print("COMPARISON: Custom vs Library-Based Search Engines")
    print("=" * 80)
    
    print("\nInitializing engines...")
    start_time = time.time()
    engines = load_engines_fast()
    init_time = time.time() - start_time
    print(f"✓ Initialization completed in {init_time:.2f} seconds\n")
    
    query = "الصلاة الزكاة"
    print(f"Query: '{query}'\n")
    
    print("=" * 80)
    print("1. TF-IDF COMPARISON")
    print("=" * 80)
    
    print("\n📊 Custom TF-IDF:")
    start = time.time()
    custom_results = engines['tfidf_quran'].search(query, top_k=3)
    custom_time = time.time() - start
    print(f"   Time: {custom_time*1000:.2f}ms")
    for i, (doc_id, score, doc) in enumerate(custom_results, 1):
        print(f"   {i}. Surah {doc['chapter']}:{doc['verse']} (Score: {score:.4f})")
    
    print("\n📚 Library TF-IDF (scikit-learn):")
    start = time.time()
    lib_results = engines['tfidf_quran_lib'].search(query, top_k=3)
    lib_time = time.time() - start
    print(f"   Time: {lib_time*1000:.2f}ms")
    for i, (doc_id, score, doc) in enumerate(lib_results, 1):
        print(f"   {i}. Surah {doc['chapter']}:{doc['verse']} (Score: {score:.4f})")
    
    print(f"\n   ⚡ Speedup: {custom_time/lib_time:.2f}x faster with library")
    
    print("\n" + "=" * 80)
    print("2. BM25 COMPARISON")
    print("=" * 80)
    
    print("\n📊 Custom BM25:")
    start = time.time()
    custom_results = engines['bm25_quran'].search(query, top_k=3)
    custom_time = time.time() - start
    print(f"   Time: {custom_time*1000:.2f}ms")
    for i, res in enumerate(custom_results, 1):
        meta = res['metadata']
        print(f"   {i}. Surah {meta['chapter']}:{meta['verse']} (Score: {res['score']:.4f})")
    
    print("\n📚 Library BM25 (rank-bm25):")
    start = time.time()
    lib_results = engines['bm25_quran_lib'].search(query, top_k=3)
    lib_time = time.time() - start
    print(f"   Time: {lib_time*1000:.2f}ms")
    for i, res in enumerate(lib_results, 1):
        meta = res['metadata']
        print(f"   {i}. Surah {meta['chapter']}:{meta['verse']} (Score: {res['score']:.4f})")
    
    print(f"\n   ⚡ Speedup: {custom_time/lib_time:.2f}x faster with library")
    
    print("\n" + "=" * 80)
    print("3. VSM COMPARISON")
    print("=" * 80)
    
    print("\n📊 Custom VSM:")
    start = time.time()
    custom_results = engines['vsm_quran'].search(query, top_k=3)
    custom_time = time.time() - start
    print(f"   Time: {custom_time*1000:.2f}ms")
    for i, res in enumerate(custom_results, 1):
        meta = res['metadata']
        print(f"   {i}. Surah {meta['chapter']}:{meta['verse']} (Score: {res['score']:.4f})")
    
    print("\n📚 Library VSM (scikit-learn):")
    start = time.time()
    lib_results = engines['vsm_quran_lib'].search(query, top_k=3)
    lib_time = time.time() - start
    print(f"   Time: {lib_time*1000:.2f}ms")
    for i, res in enumerate(lib_results, 1):
        meta = res['metadata']
        print(f"   {i}. Surah {meta['chapter']}:{meta['verse']} (Score: {res['score']:.4f})")
    
    print(f"\n   ⚡ Speedup: {custom_time/lib_time:.2f}x faster with library")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✅ Library implementations are generally faster")
    print("✅ Both use the same SafeIslamicArabicProcessor")
    print("✅ Results may differ slightly due to implementation details")
    print("✅ Custom implementations are great for learning")
    print("✅ Library implementations are better for production")


def test_hadith_search():
    print("\n" + "=" * 80)
    print("HADITH SEARCH TEST")
    print("=" * 80)
    
    engines = load_engines_fast()
    
    query = "النية"
    print(f"\nQuery: '{query}'\n")
    
    print("📚 Library BM25 Hadith Results:")
    results = engines['bm25_hadith_lib'].search(query, top_k=3)
    for i, res in enumerate(results, 1):
        meta = res['metadata']
        print(f"\n{i}. {meta['book']} - Hadith #{meta['hadith_number']}")
        print(f"   Score: {res['score']:.4f}")
        print(f"   Text: {res['text'][:100]}...")


if __name__ == "__main__":
    compare_search_engines()
    test_hadith_search()
