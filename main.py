from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from starlette.responses import JSONResponse
from data_ingestion.ingestion_pipeline import DataIngestion
from agent.workflow import GraphBuilder
from data_models.models import *
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize graph ONCE at startup — memory persists across requests
graph_service = GraphBuilder()
graph_service.build()
graph = graph_service.get_graph()

@app.get("/")
async def root():
    return FileResponse("templates/index.html")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        ingestion = DataIngestion()
        ingestion.run_pipeline(files)
        return {"message": "Files successfully processed and stored."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/query")
async def query_chatbot(request: QuestionRequest):
    try:
        messages = {"messages": [HumanMessage(content=request.question)]}
        config = {"configurable": {"thread_id": request.thread_id}}

        result = graph.invoke(messages, config=config)

        if isinstance(result, dict) and "messages" in result:
            final_output = result["messages"][-1].content
        else:
            final_output = str(result)

        return {"answer": final_output}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
