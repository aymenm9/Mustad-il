from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from preprocessing import SafeIslamicArabicProcessor


class BM25SearchEngineLib:
    def __init__(self, name: str, documents: list, processor: SafeIslamicArabicProcessor):
        self.name = name
        self.documents = documents
        self.processor = processor
        
        self.doc_id_to_idx = {}
        self.idx_to_doc = {}
        
        tokenized_corpus = []
        for idx, doc in enumerate(documents):
            tokens = doc.get('tokens', [])
            tokenized_corpus.append(tokens)
            
            if 'chapter' in doc and 'verse' in doc:
                doc_id = f"{doc['chapter']}_{doc['verse']}"
            elif 'hadith_id' in doc:
                doc_id = str(doc['hadith_id'])
            else:
                doc_id = str(idx)
            
            self.doc_id_to_idx[doc_id] = idx
            self.idx_to_doc[idx] = doc_id
        
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_result = self.processor.preprocess(query)
        query_tokens = query_result['tokens']
        
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                doc_id = self.idx_to_doc[idx]
                
                matched_tokens = [t for t in query_tokens if t in doc.get('tokens', [])]
                
                results.append({
                    'doc_id': doc_id,
                    'score': float(scores[idx]),
                    'text': doc.get('arabic_original', ''),
                    'metadata': doc,
                    'matched_tokens': matched_tokens
                })
        
        return results
