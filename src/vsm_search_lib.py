import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
from preprocessing import SafeIslamicArabicProcessor


class VectorSpaceModelLib:
    def __init__(self, name: str, documents: list, processor: SafeIslamicArabicProcessor):
        self.name = name
        self.documents = documents
        self.processor = processor
        
        self.doc_id_to_idx = {}
        self.idx_to_doc_id = {}
        
        doc_texts = []
        for idx, doc in enumerate(documents):
            tokens = doc.get('tokens', [])
            doc_texts.append(' '.join(tokens))
            
            if 'chapter' in doc and 'verse' in doc:
                doc_id = f"{doc['chapter']}_{doc['verse']}"
            elif 'hadith_id' in doc:
                doc_id = str(doc['hadith_id'])
            else:
                doc_id = str(idx)
            
            self.doc_id_to_idx[doc_id] = idx
            self.idx_to_doc_id[idx] = doc_id
        
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            lowercase=False,
            token_pattern=None
        )
        
        self.doc_vectors = self.vectorizer.fit_transform(doc_texts)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_result = self.processor.preprocess(query)
        query_tokens = query_result['tokens']
        
        if not query_tokens:
            return []
        
        query_text = ' '.join(query_tokens)
        query_vector = self.vectorizer.transform([query_text])
        
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                doc = self.documents[idx]
                doc_id = self.idx_to_doc_id[idx]
                
                results.append({
                    'doc_id': doc_id,
                    'score': float(similarities[idx]),
                    'text': doc.get('arabic_original', ''),
                    'metadata': doc
                })
        
        return results
