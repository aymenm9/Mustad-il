import math
from collections import defaultdict
from typing import List, Dict
from preprocessing import SafeIslamicArabicProcessor


class VectorSpaceModel:
    def __init__(self, name: str, inverted_index: dict, doc_metadata: dict, processor: SafeIslamicArabicProcessor):
        self.name = name
        self.inverted_index = inverted_index
        self.doc_metadata = doc_metadata
        self.processor = processor
        self.N = len(doc_metadata)
        
        self.idf = {}
        for term, data in self.inverted_index.items():
            df = data['df']
            self.idf[term] = math.log10(self.N / df) if df > 0 else 0
            
        self.doc_vectors = defaultdict(lambda: defaultdict(float))
        self.doc_norms = defaultdict(float)
        self._build_index()

    def _build_index(self):
        for term, data in self.inverted_index.items():
            curr_idf = self.idf[term]
            for doc_id, positions in data['postings'].items():
                tf = len(positions)
                tfidf = tf * curr_idf
                self.doc_vectors[doc_id][term] = tfidf
                self.doc_norms[doc_id] += tfidf ** 2
        
        for doc_id in self.doc_norms:
            self.doc_norms[doc_id] = math.sqrt(self.doc_norms[doc_id])

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_tokens = self.processor.preprocess(query)['tokens']
        if not query_tokens: 
            return []
        
        query_tf = defaultdict(int)
        for t in query_tokens: 
            query_tf[t] += 1
        
        query_vector = {}
        query_norm_sq = 0
        for term, tf in query_tf.items():
            curr_idf = self.idf.get(term, 0)
            tfidf = tf * curr_idf
            query_vector[term] = tfidf
            query_norm_sq += tfidf ** 2
            
        query_norm = math.sqrt(query_norm_sq)
        if query_norm == 0: 
            return []
        
        scores = defaultdict(float)
        for term, q_tfidf in query_vector.items():
            if term in self.inverted_index:
                for doc_id in self.inverted_index[term]['postings']:
                    d_tfidf = self.doc_vectors[doc_id][term]
                    scores[doc_id] += q_tfidf * d_tfidf
        
        results = []
        for doc_id, dot_product in scores.items():
            cosine_sim = dot_product / (query_norm * self.doc_norms[doc_id])
            
            results.append({
                'doc_id': doc_id,
                'score': cosine_sim,
                'text': self.doc_metadata[doc_id].get('arabic_original', ''),
                'metadata': self.doc_metadata[doc_id]
            })
            
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
