import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any
from preprocessing import SafeIslamicArabicProcessor


class TFIDFSearchEngineLib:
    def __init__(self, documents: list, processor: SafeIslamicArabicProcessor):
        self.documents = documents
        self.processor = processor
        self.total_docs = len(documents)
        
        self.doc_texts = [' '.join(doc['tokens']) for doc in documents]
        
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            lowercase=False,
            token_pattern=None
        )
        
        self.doc_vectors = self.vectorizer.fit_transform(self.doc_texts)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_result = self.processor.preprocess(query)
        query_text = ' '.join(query_result['tokens'])
        
        if not query_result['tokens']:
            return []
        
        query_vector = self.vectorizer.transform([query_text])
        
        scores = (self.doc_vectors * query_vector.T).toarray().flatten()
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                
                if 'chapter' in doc and 'verse' in doc:
                    doc_id = f"{doc['chapter']}_{doc['verse']}"
                elif 'hadith_id' in doc:
                    doc_id = str(doc['hadith_id'])
                else:
                    doc_id = str(idx)
                
                results.append({
                    'doc_id': doc_id,
                    'score': float(scores[idx]),
                    'text': doc.get('arabic_original', ''),
                    'metadata': doc
                })
        
        return results
