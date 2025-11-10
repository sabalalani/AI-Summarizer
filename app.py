# app.py - Flask Backend
from flask import Flask, request, jsonify, render_template
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
from flask_cors import CORS  # Needed for allowing frontend to talk to backend

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Load the Summarization Model ---
# Using a distilled BART model for faster inference
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
# Note: You can load the model on a GPU if available, e.g., device = 'cuda' or 'cpu'
device = 'cpu'
model.to(device)

# --- Define and Pre-load Available Models ---
# Key: The user-friendly name displayed on the frontend
# Value: A dictionary containing the Hugging Face model/tokenizer class and model path
AVAILABLE_MODELS = {
    "DistilBART (Fast)": {
        "path": "sshleifer/distilbart-cnn-12-6",
        "model_class": BartForConditionalGeneration,
        "tokenizer_class": BartTokenizer,
    },
    "T5-Small (Fastest)": {
        "path": "t5-small",
        "model_class": T5ForConditionalGeneration,
        "tokenizer_class": T5Tokenizer,
    },
    "BART-Large (High Quality)": {
        "path": "facebook/bart-large-cnn",
        "model_class": BartForConditionalGeneration,
        "tokenizer_class": BartTokenizer,
    }
}

# NEW: Route to serve the HTML file when the user visits the root URL
@app.route('/')
def serve_frontend():
    # Assumes index.html is in the same directory as app.py
    return render_template('index.html')


# Dictionary to hold loaded model/tokenizer objects to avoid re-loading on every request
LOADED_PIPELINES = {}


def get_summarization_pipeline(model_name):
    """Loads a model and tokenizer if not already in memory."""
    if model_name in LOADED_PIPELINES:
        return LOADED_PIPELINES[model_name]["model"], LOADED_PIPELINES[model_name]["tokenizer"]

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not recognized.")

    model_info = AVAILABLE_MODELS[model_name]

    print(f"Loading model: {model_name} from {model_info['path']}...")

    # 1. Load Tokenizer
    tokenizer = model_info["tokenizer_class"].from_pretrained(model_info["path"])

    # 2. Load Model
    model = model_info["model_class"].from_pretrained(model_info["path"])
    model.to(device)

    # 3. Store for future use
    LOADED_PIPELINES[model_name] = {"model": model, "tokenizer": tokenizer}

    return model, tokenizer


@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    input_text = data.get('text', '')
    min_length = data.get('min_length', 30)
    max_length = data.get('max_length', 150)
    # *** NEW: Get the selected model name ***
    selected_model_name = data.get('model_name', 'DistilBART (Fast)')

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # *** NEW: Get the correct model and tokenizer based on user selection ***
        model, tokenizer = get_summarization_pipeline(selected_model_name)

        # T5 models require a task prefix (important!)
        if "T5" in selected_model_name:
            input_text = "summarize: " + input_text

        # Tokenize the input text
        inputs = tokenizer([input_text], max_length=1024, return_tensors='pt', truncation=True).to(device)

        # Generate the summary
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=4,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

        return jsonify({"summary": summary})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Error during summarization: {e}")
        return jsonify({"error": "Internal server error during summarization"}), 500


if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)