from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="T5 Text Generation API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    input: str
    generated_text: str


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
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


@app.get("/")
async def home():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "T5 Model API is running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Health check for Hugging Face"""
    return {"status": "healthy"}


@app.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    """Generate text using T5 model"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        input_text = request.text

        if not input_text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Preprocess input for T5
        input_text = "summarize: " + input_text  # Example task prefix

        # Tokenize and generate
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=150,
                min_length=40,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return TextResponse(
            input=input_text,
            generated_text=generated_text
        )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)