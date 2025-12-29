from typing import List, Literal
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = "AIzaSyBBD_KmwcMCa5JIcPKaQcBAclE2u-Dh6C4"

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


def generate_strict_queries(user_question: str, api_key: str = GEMINI_API_KEY) -> List[dict]:
    if api_key is None:
        return []

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_question,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": SearchResponse.model_json_schema(),
                "system_instruction": SYSTEM_INSTRUCTION_ARABIC + "\n\nCRITICAL: Respond ONLY with the JSON object. No explanations.",
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


def validate_search_results(user_question: str, query: str, search_results: List[dict], api_key: str = GEMINI_API_KEY) -> List[dict]:
    """
    Validates the relevance of search results using a second LLM pass.
    """
    if not search_results or api_key is None:
        return []

    client = genai.Client(api_key=api_key)

    # Prepare the context for the LLM
    results_text = ""
    for i, res in enumerate(search_results):
        results_text += f"\nResult Index {i}:\n{res['text']}\n"

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
        "system_instruction": "You are an expert in Islamic text analysis. Your task is to validate if the provided Quran/Hadith search results are relevant to the user's question. For each result index, provide an observation and a relevance boolean.",
        "temperature": 0.0,
    }

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=config,
        )

        if response.text:
            result = ValidationResponse.model_validate_json(response.text)
            
            # Map back to original results
            final_results = []
            validations = {v.index: v for v in result.validated_results}
            
            for i, res in enumerate(search_results):
                if i in validations:
                    val = validations[i]
                    res_copy = res.copy()
                    res_copy["observation"] = val.observation
                    res_copy["is_relevant"] = val.is_relevant
                    final_results.append(res_copy)
                else:
                    # If LLM missed an index, assume not relevant/evaluated
                    res_copy = res.copy()
                    res_copy["observation"] = "Not evaluated by LLM"
                    res_copy["is_relevant"] = False
                    final_results.append(res_copy)
            
            return final_results

        return search_results

    except Exception as e:
        # print(f"DEBUG: Exception in validation: {e}")
        return search_results


def search_with_queries(queries: list, bm25_quran, bm25_hadith, top_k: int = 5) -> list:
    processed_queries = []
    
    for q in queries:
        query_text = q.get("query", "")
        query_type = q.get("type", "")
        
        if query_type == "quran":
            search_results = bm25_quran.search(query_text, top_k=top_k)
        elif query_type == "hadith":
            search_results = bm25_hadith.search(query_text, top_k=top_k)
        else:
            search_results = []
        
        result_obj = {
            "query": query_text,
            "type": query_type,
            "search_results": [
                {
                    "doc_id": res["doc_id"],
                    "score": res["score"],
                    "text": res["text"],
                    "metadata": res["metadata"]
                }
                for res in search_results
            ]
        }
        
        processed_queries.append(result_obj)
    
    return processed_queries
