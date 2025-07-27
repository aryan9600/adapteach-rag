import os
import logging
from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from dotenv import load_dotenv
from utils import save_and_process_pdf
from rag import answer_query, PROCESSOR, MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application.")

@app.post("/upload_pdf")
async def upload_pdf(doc_name: str = Form(...), file: UploadFile = File(...)):
    logger.info(f"Received request to upload PDF: {file.filename} for doc_name: {doc_name}")
    contents = await file.read()
    os.makedirs("uploads", exist_ok=True)
    upload_path = os.path.join("uploads", file.filename)
    logger.info(f"Saving uploaded file to: {upload_path}")
    with open(upload_path, "wb") as f:
        f.write(contents)
    
    try:
        logger.info(f"Processing PDF: {upload_path} for doc_name: {doc_name}")
        slug = save_and_process_pdf(upload_path, doc_name, PROCESSOR, MODEL)
        logger.info(f"Successfully processed and stored document with slug: {slug}")
    except Exception as e:
        logger.error(f"Error processing PDF for doc_name: {doc_name}. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"doc_name": slug}

@app.post("/query")
async def query(request: Request):
    try:
        body = await request.json()
        doc_slug = body.get("doc_slug")
        query = body.get("query")
        top_k = body.get("top_k", 0)
        logger.info(f"Received query for doc_name: {doc_slug} with query: '{query}' and top_k: {top_k}")

        result = answer_query(doc_slug, query, top_k)
        logger.info(f"Successfully answered query for doc_name: {doc_slug}")
    except ValueError as e:
        logger.warning(f"ValueError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
        
    return result

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server.")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
