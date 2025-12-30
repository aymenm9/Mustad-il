from typing import List, Literal, Union
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class SearchQuery(BaseModel):
    query: str = Field(description="Explicit Arabic search phrase.")
    type: Literal["quran", "hadith"] = Field(description="The target source text.")

class SearchResponse(BaseModel):
    queries: List[SearchQuery]

class ValidationResult(BaseModel):
    index: int = Field(description="The index of the result in the provided list.")
    observation: str = Field(description="Brief observation about why the result is relevant or not.")
    is_relevant: bool = Field(description="Whether the result directly addresses the user question.")

class ValidationResponse(BaseModel):
    validated_results: List[ValidationResult]

SYSTEM_INSTRUCTION_ARABIC = """
أنت خبير تقني في استرجاع المعلومات من النصوص الإسلامية (القرآن الكريم والحديث الشريف). مهمتك هي تحويل سؤال المستخدم إلى استعلامات بحث "مستهدفة" (Targeted Queries) بدقة عالية.

الرسم العثماني (Quranic Script):
يجب أن تكون استعلامات القرآن الكريم حصراً بـ "الرسم العثماني" الأصيل كما ورد في المصحف.
1. استخدم الإملاء القرآني: (الصلوة، الزكوة، الحيوة، السَّمٰوٰت، يٰأَيُّهَا، كِتٰب).
2. بالنسبة للقرآن: يُفضل بشدة استخدام التشكيل الكامل (Diacritics) والعلامات الخاصة بالرسم العثماني (مثل الألف الخنجرية ٰ).
3. بالنسبة للحديث: استخدم الإملاء الحديث مع التشكيل أو بدونه.

استراتيجية الاستهداف (Targeting Strategy):
بدلاً من الكلمات العامة، ولّد عبارات نصية كاملة أو مقاطع من الآيات والأحاديث التي تعالج الموضوع.
1. تجنب الكلمات المفردة الضعيفة (مثل: "الله"، "السماء"، "الارض") إلا إذا كانت جزءاً من سياق فريد.
2. ولّد جمل بحثية (3-6 كلمات) تمثل نصاً محتملاً في المصدر.
3. نوع في الاستعلامات لتشمل:
   - عبارات صريحة (Direct Phrasing).
   - مرادفات لفظية قرآنية ونبوية.
   - الكلمات المفتاحية التقنية (Technical Keywords) الموجودة في المتون.

مثال: سؤل "أين الله"؟
الاستعلامات المستهدفة:
- [quran] "الرَّحْمٰنُ عَلَى الْعَرْشِ اسْتَوَىٰ"
- [quran] "أَأَمِنْتُمْ مَنْ فِي السَّمَاءِ"
- [quran] "وَهُوَ مَعَكُمْ أَيْنَ مَا كُنْتُمْ"
- [hadith] "أين الله؟ قالت: في السماء"
- [hadith] "ينزل ربنا تبارك وتعالى كل ليلة إلى السماء الدنيا"

القيود الصارمة (STRICT CONSTRAINTS):
1. استعلامات القرآن: يجب أن تلتزم بالرسم العثماني (مثلاً: "الصلوة" وليس "الصلاة" إذا كانت للقرآن).
2. لا تقم بالإجابة على السؤال أو شرحه.
3. لا تولد مصطلحات فقهية معاصرة غير موجودة في المتن (مثل: "فقه المعاملات").
4. المخرج يجب أن يكون قائمة من الكائنات تحتوي على (query) و (type).

ملاحظة تقنية: النظام سيقوم بمعالجة نصوصك آلياً، لذا فإن استخدامك للرسم العثماني والتشكيل سيساعد في مطابقة الأنماط العميقة للنصوص.
"""

SYSTEM_INSTRUCTION_VALIDATION_ARABIC = """
أنت خبير في تحليل النصوص الإسلامية. مهمتك هي التحقق مما إذا كانت نتائج البحث المقدمة (القرآن/الحديث) ذات صلة بسؤال المستخدم. لكل نتيجة، قدم ملاحظة وتقييماً للملاءمة (صواب/خطأ).

تعليمات:
1. اقرأ سؤال المستخدم بعناية.
2. افحص نتيجة البحث المقدمة.
3. حدد ما إذا كانت النتيجة تجيب على السؤال أو تتعلق به بشكل مباشر.
4. "is_relevant": يجب أن يكون true فقط إذا كانت النتيجة ذات صلة ومفيدة.
5. "observation": اشرح سبب حكمك باختصار باللغة العربية.
"""


class SearchModelOne:
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def generate_queries(self, user_question: str) -> List[dict]:
        """
        Generates targeted search queries from a user question.
        Returns a list of dicts with keys 'query' and 'type'.
        """
        if not self.client:
            return []

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=user_question,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": SearchResponse.model_json_schema(),
                    "system_instruction": SYSTEM_INSTRUCTION_ARABIC,
                    "temperature": 0.0,
                },
            )

            if response.text:
                result = SearchResponse.model_validate_json(response.text)
                
                # Convert Pydantic objects to dicts and sort
                queries_list = [q.model_dump() for q in result.queries]
                
                return sorted(
                    queries_list, 
                    key=lambda x: 0 if x.get("type") == "quran" else 1
                )

            return []

        except Exception as e:
            print(f"DEBUG Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def filter_results(self, user_question: str, query: str, results: List[str]) -> ValidationResponse:
        """
        Validates the relevance of search results (text strings).
        Returns only the strings that are relevant to the user's question.
        """
        if not results or not self.client:
            return []

        results_text = ""
        for i, text in enumerate(results):
            clean_text = text[:1000] if isinstance(text, str) else str(text)[:1000]
            results_text += f"\nResult Index {i}:\n{clean_text}\n"

        prompt = f"""
User Question: {user_question}
Search Query used: {query}

Please evaluate the following search results for relevance to the User Question.
Results to evaluate:
{results_text}
"""

        config = {
            "response_mime_type": "application/json",
            "response_json_schema": ValidationResponse.model_json_schema(),
            "system_instruction": SYSTEM_INSTRUCTION_VALIDATION_ARABIC,
            "temperature": 0.0,
        }

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=config,
            )

            if response.text:
                result = ValidationResponse.model_validate_json(response.text)
                return result.validated_results

            return []

        except Exception as e:
            print(f"DEBUG: Exception in validation: {e}")
            return [] 


class SearchModelTwo:
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            self.model = self.client.models
        else:
            self.client = None
            self.model = None

    def generate_queries(self, user_question: str) -> List[dict]:
        """
        Generates simple phrase-based search queries.
        Returns a list of dicts with keys 'query' and 'type'.
        """
        if not self.model:
            return []
        
        queries = []
        
        quran_phrases = self._generate_quran(user_question)
        for phrase in quran_phrases:
            queries.append({"query": phrase, "type": "quran"})
        
        hadith_phrases = self._generate_hadith(user_question)
        for phrase in hadith_phrases:
            queries.append({"query": phrase, "type": "hadith"})
        
        return queries

    def _generate_quran(self, question: str) -> List[str]:
        prompt = f"""
        أنت عالم قرآني متخصص.
        السؤال: {question}
        أنتج 8-12 عبارة قصيرة (2-7 كلمات) موجودة حرفيًا في القرآن الكريم تتعلق بالموضوع.
        أعطِ العبارات فقط، كل في سطر، بدون أي إضافات.
        """
        try:
            response = self.model.generate_content(model=MODEL_NAME, contents=prompt)
            text = response.text.strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            return [l for l in lines if 2 <= len(l.split()) <= 8][:12]
        except:
            return []

    def _generate_hadith(self, question: str) -> List[str]:
        prompt = f"""
        أنت عالم حديث متخصص.
        السؤال: {question}
        أنتج 8-12 عبارة قصيرة (2-8 كلمات) موجودة حرفيًا في الأحاديث الصحيحة تتعلق بالموضوع.
        أعطِ العبارات فقط، كل في سطر، بدون أي إضافات.
        """
        try:
            response = self.model.generate_content(model=MODEL_NAME, contents=prompt)
            text = response.text.strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            return [l for l in lines if 2 <= len(l.split()) <= 9][:12]
        except:
            return []
