from fastapi import FastAPI
from load_engines import load_engines_fast
from gemini_llm import GeminiSearchModel
from run_user_query import run_query, SearchEngine
from schemas import AppSearchResponse

app = FastAPI()
engines = {}
llm_model = GeminiSearchModel()

class AppSearchEngine(SearchEngine):
    def __init__(self, engines, algorithm="bm25"):
        self.engines = engines
        self.algorithm = algorithm

    def search(self, query: str, type: str, top_k: int = 5):
        name = f"{self.algorithm}_{type}"
        engine = self.engines.get(name)
        return engine.search(query, top_k=top_k) if engine else []

@app.on_event("startup")
async def startup_event():
    global engines
    engines = load_engines_fast()

@app.get("/search/{query}")
@app.get("/search/{algorithm}/{model}/{query}")
def search(query: str, algorithm: str = "bm25", model: str = "gemini"):
    engine = AppSearchEngine(engines, algorithm)
    return run_query(query, engine, llm_model)

if __name__ == "__main__":
    import uvicorn
    engines = load_engines_fast()
    uvicorn.run(app, host="0.0.0.0", port=8000)
