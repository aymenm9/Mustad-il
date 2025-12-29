from preprocessing import SafeIslamicArabicProcessor
from indexing import (
    build_quran_index,
    build_hadith_index,
    save_index,
    build_inverted_index_quran,
    build_inverted_index_hadith,
    save_inverted_index
)
from tfidf_search import TFIDFSearchEngine
from bm25_search import BM25SearchEngine
from vsm_search import VectorSpaceModel
from tfidf_search_lib import TFIDFSearchEngineLib
from bm25_search_lib import BM25SearchEngineLib
from vsm_search_lib import VectorSpaceModelLib
from gemini_llm import generate_strict_queries, search_with_queries


def main():
    processor = SafeIslamicArabicProcessor()
    
    quran_index = build_quran_index('qoran/quran.json', processor)
    hadith_index = build_hadith_index('hadith', processor)
    
    save_index(quran_index, "quran")
    save_index(hadith_index, "hadith")
    
    quran_inverted_index = build_inverted_index_quran(quran_index)
    hadith_inverted_index = build_inverted_index_hadith(hadith_index)
    
    save_inverted_index(quran_inverted_index, "quran")
    save_inverted_index(hadith_inverted_index, "hadith")
    
    quran_dict = {f"{r['chapter']}_{r['verse']}": r for r in quran_index}
    hadith_dict = {str(r['hadith_id']): r for r in hadith_index}
    
    tfidf_quran = TFIDFSearchEngine(
        inverted_index=quran_inverted_index,
        documents=quran_index,
        total_docs=len(quran_index),
        processor=processor
    )
    
    tfidf_hadith = TFIDFSearchEngine(
        inverted_index=hadith_inverted_index,
        documents=hadith_index,
        total_docs=len(hadith_index),
        processor=processor
    )
    
    bm25_quran = BM25SearchEngine(
        name="Quran",
        inverted_index=quran_inverted_index,
        doc_metadata=quran_dict,
        processor=processor
    )
    
    bm25_hadith = BM25SearchEngine(
        name="Hadith",
        inverted_index=hadith_inverted_index,
        doc_metadata=hadith_dict,
        processor=processor
    )
    
    vsm_quran = VectorSpaceModel(
        name="Quran",
        inverted_index=quran_inverted_index,
        doc_metadata=quran_dict,
        processor=processor
    )
    
    vsm_hadith = VectorSpaceModel(
        name="Hadith",
        inverted_index=hadith_inverted_index,
        doc_metadata=hadith_dict,
        processor=processor
    )
    
    tfidf_quran_lib = TFIDFSearchEngineLib(
        documents=quran_index,
        processor=processor
    )
    
    tfidf_hadith_lib = TFIDFSearchEngineLib(
        documents=hadith_index,
        processor=processor
    )
    
    bm25_quran_lib = BM25SearchEngineLib(
        name="Quran",
        documents=quran_index,
        processor=processor
    )
    
    bm25_hadith_lib = BM25SearchEngineLib(
        name="Hadith",
        documents=hadith_index,
        processor=processor
    )
    
    vsm_quran_lib = VectorSpaceModelLib(
        name="Quran",
        documents=quran_index,
        processor=processor
    )
    
    vsm_hadith_lib = VectorSpaceModelLib(
        name="Hadith",
        documents=hadith_index,
        processor=processor
    )
    
    return {
        'processor': processor,
        'quran_index': quran_index,
        'hadith_index': hadith_index,
        'quran_inverted_index': quran_inverted_index,
        'hadith_inverted_index': hadith_inverted_index,
        'tfidf_quran': tfidf_quran,
        'tfidf_hadith': tfidf_hadith,
        'bm25_quran': bm25_quran,
        'bm25_hadith': bm25_hadith,
        'vsm_quran': vsm_quran,
        'vsm_hadith': vsm_hadith,
        'tfidf_quran_lib': tfidf_quran_lib,
        'tfidf_hadith_lib': tfidf_hadith_lib,
        'bm25_quran_lib': bm25_quran_lib,
        'bm25_hadith_lib': bm25_hadith_lib,
        'vsm_quran_lib': vsm_quran_lib,
        'vsm_hadith_lib': vsm_hadith_lib
    }


if __name__ == "__main__":
    engines = main()
