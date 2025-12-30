import math
from typing import List, Tuple, Dict, Any
from preprocessing import SafeIslamicArabicProcessor


class TFIDFSearchEngine:
    def __init__(self, inverted_index: dict, documents: list, total_docs: int, processor: SafeIslamicArabicProcessor):
        self.inverted_index = inverted_index
        self.documents = documents
        self.total_docs = total_docs
        self.processor = processor
    
    def calculate_tf(self, term_freq: int, doc_length: int) -> float:
        if doc_length == 0:
            return 0.0
        return term_freq / doc_length
    
    def calculate_idf(self, df: int) -> float:
        if df == 0:
            return 0.0
        return math.log10(self.total_docs / df)
    
    def calculate_tfidf(self, tf: float, idf: float) -> float:
        return tf * idf
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_result = self.processor.preprocess(query)
        query_terms = query_result['tokens']
        
        if not query_terms:
            return []
        
        doc_scores = {}
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            term_data = self.inverted_index[term]
            df = term_data['df']
            idf = self.calculate_idf(df)
            
            for doc_id, positions in term_data['postings'].items():
                doc = None
                for d in self.documents:
                    if hasattr(d, 'get'):
                        if 'chapter' in d and 'verse' in d:
                            if doc_id == f"{d['chapter']}_{d['verse']}":
                                doc = d
                                break
                        elif 'hadith_id' in d:
                            if doc_id == str(d['hadith_id']):
                                doc = d
                                break
                
                if doc is None:
                    continue
                
                term_freq = len(positions)
                doc_length = len(doc['tokens'])
                tf = self.calculate_tf(term_freq, doc_length)
                
                tfidf = self.calculate_tfidf(tf, idf)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'score': 0.0, 'doc': doc}
                doc_scores[doc_id]['score'] += tfidf
        
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k]
        
        results = []
        for doc_id, data in ranked_docs:
            doc = data['doc']
            results.append({
                'doc_id': doc_id,
                'score': float(data['score']),
                'text': doc.get('arabic_original', ''),
                'metadata': doc
            })
        return results
