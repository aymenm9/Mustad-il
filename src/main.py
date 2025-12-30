from fastapi import FastAPI
from load_engines import load_engines_fast
from gemini_llm import SearchModelOne, SearchModelTwo
from run_user_query import run_query, run_query_model_two
from schemas import AppSearchResponse

app = FastAPI()
engines = {}
llm_model = {
    "m1": SearchModelOne(),
    "m2": SearchModelTwo()
}



@app.on_event("startup")
async def startup_event():
    global engines
    engines = load_engines_fast()


@app.get("/search/{engine}/{model}/{query}")
def search(query: str, engine: str = "bm25", model: str = "m1") -> AppSearchResponse:
    engine_quran = engines.get(f'{engine}_quran', engines.get('bm25_quran'))
    engine_hadith = engines.get(f'{engine}_hadith', engines.get('bm25_hadith'))
    
    selected_model = llm_model.get(model, llm_model["m1"])
    
    if model == "m2":
        return run_query_model_two(query, engine_quran, engine_hadith, selected_model)
    else:
        return run_query(query, engine_quran, engine_hadith, selected_model)

if __name__ == "__main__":
    import uvicorn
    engines = load_engines_fast()
    uvicorn.run(app, host="0.0.0.0", port=8000)
