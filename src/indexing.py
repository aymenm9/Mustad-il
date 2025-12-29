import json
import os
from typing import List, Dict
from collections import defaultdict
import pandas as pd
from preprocessing import SafeIslamicArabicProcessor


def build_quran_index(quran_path: str, processor: SafeIslamicArabicProcessor) -> List[Dict]:
    with open(quran_path, 'r', encoding='utf-8') as f:
        quran_data = json.load(f)

    quran_index = []

    if isinstance(quran_data, list):
        verses_iter = quran_data
    elif isinstance(quran_data, dict):
        verses_iter = []
        for verses in quran_data.values():
            verses_iter.extend(verses)
    else:
        raise ValueError("Unsupported Quran JSON structure")

    for verse in verses_iter:
        pre = processor.preprocess(verse['text'])

        record = {
            'chapter': verse.get('chapter'),
            'verse': verse.get('verse'),
            'arabic_original': pre['original'],
            'arabic_clean': pre['clean'],
            'arabic_normalized': pre['normalized'],
            'tokens': pre['tokens']
        }

        quran_index.append(record)

    return quran_index


def build_hadith_index(hadith_folder: str, processor: SafeIslamicArabicProcessor) -> List[Dict]:
    book_names = {
        'bukhari': {'ar': 'صحيح البخاري', 'en': 'Sahih al-Bukhari'},
        'muslim': {'ar': 'صحيح مسلم', 'en': 'Sahih Muslim'},
        'malik': {'ar': 'موطأ مالك', 'en': 'Muwatta Malik'}
    }

    hadith_index = []

    for book_key, names in book_names.items():
        file_path = os.path.join(hadith_folder, f"{book_key}.json")

        if not os.path.exists(file_path):
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for h in data.get('hadiths', []):
            arabic_text = h.get('arabic')
            if not arabic_text:
                continue

            pre = processor.preprocess(arabic_text)

            english = h.get('english')
            if isinstance(english, dict):
                english_text = english.get('text')
                narrator = english.get('narrator')
            else:
                english_text = None
                narrator = None

            record = {
                'book': names['en'],
                'book_ar': names['ar'],
                'hadith_id': h.get('id'),
                'hadith_number': h.get('idInBook'),
                'chapter_id': h.get('chapterId'),
                'arabic_original': pre['original'],
                'arabic_clean': pre['clean'],
                'arabic_normalized': pre['normalized'],
                'tokens': pre['tokens'],
                'english_text': english_text,
                'narrator': narrator
            }

            hadith_index.append(record)

    return hadith_index


def save_index(index_data: list, name: str, output_dir: str = 'indices'):
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = f"{output_dir}/{name}_index.json"
    csv_path = f"{output_dir}/{name}_index.csv"
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    df = pd.DataFrame(index_data)
    if 'tokens' in df.columns:
        df['tokens'] = df['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
    df.to_csv(csv_path, index=False, encoding='utf-8')


def build_inverted_index_quran(quran_index_data: list) -> dict:
    inverted_index = defaultdict(lambda: {'df': 0, 'postings': defaultdict(list)})

    for record in quran_index_data:
        doc_id = f"{record['chapter']}_{record['verse']}"
        tokens = record.get('tokens', [])
        seen_terms = set()

        for position, term in enumerate(tokens):
            if not term:
                continue

            inverted_index[term]['postings'][doc_id].append(position)

            if term not in seen_terms:
                inverted_index[term]['df'] += 1
                seen_terms.add(term)

    final_index = {}
    for term, data in inverted_index.items():
        final_index[term] = {
            'df': data['df'],
            'postings': dict(data['postings'])
        }

    return final_index


def build_inverted_index_hadith(hadith_index_data: list) -> dict:
    inverted_index = defaultdict(lambda: {'df': 0, 'postings': defaultdict(list)})

    for record in hadith_index_data:
        doc_id = str(record.get('hadith_id', 'unknown'))
        tokens = record.get('tokens', [])
        seen_terms = set()

        for position, term in enumerate(tokens):
            if not term:
                continue

            inverted_index[term]['postings'][doc_id].append(position)

            if term not in seen_terms:
                inverted_index[term]['df'] += 1
                seen_terms.add(term)

    final_index = {}
    for term, data in inverted_index.items():
        final_index[term] = {
            'df': data['df'],
            'postings': dict(data['postings'])
        }

    return final_index


def save_inverted_index(inverted_index: dict, name: str, output_dir: str = 'indices'):
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = f'{output_dir}/{name}_inverted_index.json'
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, ensure_ascii=False, indent=2)
