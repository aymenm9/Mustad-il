import re
import unicodedata
from typing import List, Dict, Any


class SafeIslamicArabicProcessor:
    def __init__(self):
        self.protected_terms = {
            'الله', 'الرحمن', 'الرحيم', 'الملك', 'القدوس', 'السلام',
            'المؤمن', 'المهيمن', 'العزيز', 'الجبار', 'المتكبر',
            'الخالق', 'البارئ', 'المصور', 'الغفار', 'القهار',
            'الوهاب', 'الرزاق', 'الفتاح', 'العليم',
            'السميع', 'البصير', 'الحكم', 'العدل',
            'اللطيف', 'الخبير', 'الحليم', 'العظيم',
            'الغفور', 'الشكور', 'العلي', 'الكبير',
            'الإسلام', 'الإيمان', 'الإحسان',
            'الصلاة', 'الزكاة', 'الصيام', 'الحج', 'العمرة',
            'القرآن', 'الوحي',
            'محمد', 'النبي', 'الرسول',
            'إبراهيم', 'موسى', 'عيسى', 'نوح', 'يوسف',
            'القدر', 'الآخرة', 'الجنة', 'النار',
            'الكعبة', 'مكة', 'المدينة',
            'الجمعة'
        }

        self.protected_phrases = {
            'رسول الله',
            'بيت الله',
            'عبد الله',
            'رضي الله عنه',
            'صلى الله عليه وسلم'
        }

    def remove_diacritics(self, text: str) -> str:
        text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]', '', text)
        normalized = unicodedata.normalize('NFKD', text)
        return ''.join(c for c in normalized if not unicodedata.combining(c))

    def normalize(self, text: str) -> str:
        text = re.sub(r'[إأآٱ]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ـ+', '', text)
        return text

    def protect_phrases(self, text: str) -> str:
        for p in self.protected_phrases:
            safe = p.replace(' ', '_')
            text = text.replace(p, safe)
        return text

    def restore_phrases(self, tokens: List[str]) -> List[str]:
        return [t.replace('_', ' ') for t in tokens]

    def tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\u0621-\u064A\s_]', ' ', text)
        return [t for t in text.split() if t]

    def preprocess(self, text: str) -> Dict[str, Any]:
        original = text
        text = self.protect_phrases(text)
        text = self.remove_diacritics(text)
        text = self.normalize(text)
        tokens = self.tokenize(text)
        tokens = self.restore_phrases(tokens)

        return {
            'original': original,
            'normalized': text,
            'tokens': tokens,
            'clean': ' '.join(tokens)
        }
