from preprocessing import SafeIslamicArabicProcessor
from indexing import (
    build_quran_index,
    build_hadith_index,
    save_index,
    build_inverted_index_quran,
    build_inverted_index_hadith,
    save_inverted_index
)

def build_indices():
    """
    Build and save all indices for Quran and Hadith.
    This includes the forward indices (documents) and inverted indices.
    """
    processor = SafeIslamicArabicProcessor()
    
    quran_index = build_quran_index('qoran/quran.json', processor)
    
    hadith_index = build_hadith_index('hadith', processor)
    
 
    save_index(quran_index, "quran")
    save_index(hadith_index, "hadith")

    quran_inverted_index = build_inverted_index_quran(quran_index)
    hadith_inverted_index = build_inverted_index_hadith(hadith_index)

    save_inverted_index(quran_inverted_index, "quran")
    save_inverted_index(hadith_inverted_index, "hadith")
    

if __name__ == "__main__":
    build_indices()
