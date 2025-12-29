import math
from collections import defaultdict
from typing import List, Dict, Any
from preprocessing import SafeIslamicArabicProcessor


class BM25SearchEngine:
    def __init__(self, name: str, inverted_index: Dict[str, Any], doc_metadata: Dict[str, Any], 
                 processor: SafeIslamicArabicProcessor, k1: float = 1.5, b: float = 0.75):
        self.name = name
        self.inverted_index = inverted_index
        self.doc_metadata = doc_metadata
        self.processor = processor
        self.k1 = k1
        self.b = b
        
        self.N = len(doc_metadata)
        self.avg_dl = sum(len(d.get('tokens', [])) for d in doc_metadata.values()) / self.N
    
    def _calculate_idf(self, df: int) -> float:
        return math.log(((self.N - df + 0.5) / (df + 0.5)) + 1)
    
    def _score_bm25(self, tf: int, doc_len: int, idf: float) -> float:
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_dl))
        return idf * (numerator / denominator)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_result = self.processor.preprocess(query)
        query_tokens = query_result['tokens']
        
        scores = defaultdict(float)
        matched_tokens = defaultdict(set)
        
        for token in query_tokens:
            if token in self.inverted_index:
                idx_entry = self.inverted_index[token]
                df = idx_entry['df']
                idf = self._calculate_idf(df)
                
                postings = idx_entry['postings']
                for doc_id, positions in postings.items():
                    tf = len(positions)
                    doc_len = len(self.doc_metadata[doc_id].get('tokens', []))
                    
                    score = self._score_bm25(tf, doc_len, idf)
                    scores[doc_id] += score
                    matched_tokens[doc_id].add(token)
        
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: (len(matched_tokens[x[0]]), x[1]),
            reverse=True
        )[:top_k]
        
        results = []
        for doc_id, score in sorted_docs:
            doc_info = self.doc_metadata[doc_id]
            results.append({
                'doc_id': doc_id,
                'score': score,
                'text': doc_info.get('arabic_original', ''),
                'metadata': doc_info,
                'matched_tokens': list(matched_tokens[doc_id])
            })
        
        return results
