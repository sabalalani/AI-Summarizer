from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
import uvicorn
import os

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer
    try:
        logger.info("Loading T5 model and tokenizer...")
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down application...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="AI Text Summarizer",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount templates directory
app.mount("/templates", StaticFiles(directory="templates"), name="templates")


class SummarizeRequest(BaseModel):
    text: str
    min_length: int = 30
    max_length: int = 150
    model_name: str = "T5-Small (Fastest)"


class SummarizeResponse(BaseModel):
    summary: str


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML from templates directory"""
    return FileResponse('templates/index.html')


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """Summarize text using T5 model"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        input_text = request.text

        if not input_text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Preprocess input for T5 - add "summarize:" prefix
        input_text = "summarize: " + input_text

        # Tokenize
        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=request.max_length,
                min_length=request.min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return SummarizeResponse(summary=summary)

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)