from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from graphs import graphs_router


app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return FileResponse('static/index.html')

app.include_router(graphs_router)
