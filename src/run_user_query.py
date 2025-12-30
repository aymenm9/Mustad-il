from typing import List, Dict, Any
from schemas import AppSearchResponse, SearchResultItem, QuranMetadata, HadithMetadata
from gemini_llm import GeminiSearchModel

class SearchEngine:
    def search(self, query: str, type: str, top_k: int = 5) -> List[Dict]:
        pass

def run_query(user_question: str, engine: SearchEngine, model: GeminiSearchModel) -> AppSearchResponse:
    queries = model.generate_queries(user_question)
    final_results = []
    seen_texts = set()

    for q in queries:
        raw_results = engine.search(q['query'], q['type'], top_k=5)
        texts = [r['text'] for r in raw_results]
        validations = model.filter_results(user_question, q['query'], texts)
        
        for val in validations:
            if val.index < 0 or val.index >= len(raw_results):
                continue
            if val.is_relevant:
                res = raw_results[val.index]
                if res['text'] in seen_texts:
                    continue
                seen_texts.add(res['text'])
                meta_raw = res['metadata']
                if 'chapter' in meta_raw:
                    meta = QuranMetadata(chapter=meta_raw['chapter'], verse=meta_raw['verse'])
                else:
                    meta = HadithMetadata(
                        book=meta_raw.get('book', ''),
                        hadith_number=meta_raw.get('hadith_number', 0),
                        hadith_id=meta_raw.get('hadith_id')
                    )
                item = SearchResultItem(
                    text=res['text'],
                    metadata=meta,
                    score=res.get('score'),
                    is_relevant=True,
                    observation=val.observation
                )
                final_results.append(item)

    return AppSearchResponse(
        user_question=user_question,
        generated_queries=queries,
        results=final_results
    )