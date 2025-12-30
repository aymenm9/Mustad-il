from typing import List, Literal, Union
import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "gemini-2.5-flash-lite"
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
أنت خبير فني متقدم في استرجاع المعلومات من "متون" النصوص الإسلامية (القرآن الكريم والحديث الشريف). مهمتك هي استخراج "عبارات دلالية" و "نصوص مرتبطة" من صلب المصادر، وليس تصنيفها.

⚠️ تحذير صارم (🚫 ممنوع تماماً):
- لا تولد أسماء كتب (مثل: أصول الدين، صحيح البخاري، فقه العبادات).
- لا تولد مصطلحات فقهية أو عقدية معاصرة (مثل: عقيدة أهل السنة، مسائل الإيمان، التوحيد).
- لا تولد كلمات تصنيفية (مثل: باب، فصل، كتاب، مبحث).

✅ المطلوب (الاسترجاع الدلالي والنصي):
1. ولّد عبارات تمثل "نصاً محتملاً" أو "صياغة بديلة" موجودة في القرآن أو الحديث (مثل: "الرَّحْمٰنُ عَلَى الْعَرْشِ اسْتَوَىٰ" أو "غمرت السماء" كإشارات للمكانية).
2. استخدم الكلمات المفتاحية "الأصيلة" ومرادفاتها القرآنية (مثل: "البرية"، "الخلق"، "القيامة"، "الصلاة").
3. لا تقتصر على المطابقة الحرفية الصرفة؛ ابحث عن العبارات التي تحمل "جوهر" المعنى في لغة النص الأصلي.
4. بالنسبة للقرآن: يفضل استخدام "الرسم العثماني" والتشكيل (السَّمٰوٰت، يٰأَيُّهَا، كِتٰب).
5. بالنسبة للحديث: ولّد مقاطع تعبيرية من المتون (مثل: "بني الإسلام على خمس" أو "بنيان مرصوص").

الاستراتيجية:
تخيل أنك تبحث عن "أثر لفظي أو معنوي" داخل النص. الاستعلامات يجب أن تكون جملًا أو كلمات مفتاحية تعبر عن الموضوع كما ورد في زمن النص، وليست عناوينًا حديثة.
"""

SYSTEM_INSTRUCTION_VALIDATION_ARABIC = """
أنت خبير في تحليل النصوص الإسلامية. مهمتك هي تقييم مدى صلة نتائج البحث (القرآن/الحديث) بسؤال المستخدم.

معايير القبول:
1. اقبل النتيجة إذا كانت تتعلق بالموضوع العام للسؤال، حتى لو لم تجب عليه مباشرة.
2. اقبل النتيجة إذا كانت تتناول أحد جوانب الموضوع أو تذكر مفاهيم ذات صلة.
3. اقبل الآيات القرآنية والأحاديث التي قد يستدل بها في الموضوع.
4. ارفض النتائج التي لا علاقة لها بالموضوع على الإطلاق.

تعليمات:
- "is_relevant": ضع true إذا كانت النتيجة مرتبطة بالموضوع.
- "observation": اشرح العلاقة بين النتيجة والسؤال باختصار.
"""


class SearchModelOne:
    def __init__(self, api_key: str = GEMINI_API_KEY, validation_model: str = "gemini-2.5-flash-lite"):
        self.api_key = api_key
        self.validation_model = validation_model
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

    def filter_results_batch(self, user_question: str, query_results_map: List[dict]) -> dict:
        """
        Validates all search results in a single API call.
        
        Args:
            user_question: The original user question
            query_results_map: List of dicts with 'query', 'type', and 'results' keys
                              where 'results' is a list of result dicts
        
        Returns:
            Dict mapping query indices to lists of ValidationResult objects
        """
        if not query_results_map or not self.client:
            return {}

        # Build comprehensive prompt with all queries and results
        all_results_text = ""
        for q_idx, item in enumerate(query_results_map):
            query = item['query']
            query_type = item['type']
            results = item['results']
            
            all_results_text += f"\n{'='*60}\n"
            all_results_text += f"Query {q_idx} [{query_type}]: {query}\n"
            all_results_text += f"{'='*60}\n"
            
            for r_idx, result in enumerate(results):
                text = result.get('text', '')
                clean_text = text[:1000] if isinstance(text, str) else str(text)[:1000]
                all_results_text += f"\nQuery {q_idx}, Result {r_idx}:\n{clean_text}\n"

        prompt = f"""
User Question: {user_question}

Please evaluate ALL the following search results for relevance to the User Question.
For each result, provide the query index and result index along with your evaluation.

{all_results_text}

IMPORTANT: Return validations for ALL results shown above. Use the format "Query X, Result Y" indices.
"""

        # Extended validation schema to include query_index
        class ExtendedValidationResult(BaseModel):
            query_index: int = Field(description="The query index (Query X)")
            result_index: int = Field(description="The result index within that query (Result Y)")
            observation: str = Field(description="Brief observation about relevance")
            is_relevant: bool = Field(description="Whether the result is relevant")

        class ExtendedValidationResponse(BaseModel):
            validated_results: List[ExtendedValidationResult]

        config = {
            "response_mime_type": "application/json",
            "response_json_schema": ExtendedValidationResponse.model_json_schema(),
            "system_instruction": SYSTEM_INSTRUCTION_VALIDATION_ARABIC,
            "temperature": 0.0,
        }

        try:
            response = self.client.models.generate_content(
                model=self.validation_model,
                contents=prompt,
                config=config,
            )

            if response.text:
                result = ExtendedValidationResponse.model_validate_json(response.text)
                
                # Organize results by query index
                validations_by_query = {}
                for val in result.validated_results:
                    q_idx = val.query_index
                    if q_idx not in validations_by_query:
                        validations_by_query[q_idx] = []
                    validations_by_query[q_idx].append({
                        'index': val.result_index,
                        'observation': val.observation,
                        'is_relevant': val.is_relevant
                    })
                
                return validations_by_query

            return {}

        except Exception as e:
            print(f"DEBUG: Exception in batch validation: {e}")
            import traceback
            traceback.print_exc()
            return {} 


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
        أنت باحث في مفردات ومعاني القرآن الكريم.
        السؤال: {question}
        المطلوب: توليد 8-12 عبارة (مقاطع من آيات أو كلمات مفتاحية قرآنية) تتعلق بالموضوع دلالياً أو نصياً.
        🚫 ممنوع: العناوين (مثل: عقيدة)، أو أسماء السور، أو المصطلحات الحديثة.
        ✅ المطلوب: عبارات تعكس "الجوهر القرآني" للموضوع (مثل: "فبأي آلاء ربكما تكذبان" أو "خلق الإنسان من علق").
        أعطِ العبارات فقط، كل في سطر.
        """
        try:
            response = self.model.generate_content(model=MODEL_NAME, contents=prompt)
            text = response.text.strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            return [l for l in lines if 2 <= len(l.split()) <= 10][:12]
        except:
            return []

    def _generate_hadith(self, question: str) -> List[str]:
        prompt = f"""
        أنت خبير في متون معاني الحديث الشريف.
        السؤال: {question}
        المطلوب: توليد 8-12 عبارة (مقاطع من المتون أو عبارات نبوية شائعة) ترتبط بالموضوع دلالياً.
        🚫 ممنوع: أسماء الكتب (صحيح البخاري)، أو التصنيفات الفقهية (كتاب الصلاة)، أو لغة الفقهاء المتأخرين.
        ✅ المطلوب: لغة النبوة والحكمة (مثل: "كلكم راع" أو "المرء مع من أحب").
        أعطِ العبارات فقط، كل في سطر.
        """
        try:
            response = self.model.generate_content(model=MODEL_NAME, contents=prompt)
            text = response.text.strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            return [l for l in lines if 2 <= len(l.split()) <= 10][:12]
        except:
            return []
