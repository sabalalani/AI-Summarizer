from flask import Flask, request, jsonify, render_template
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
from flask_cors import CORS

app = Flask(__name__, template_folder='.')
CORS(app)
# Explicitly setting to 'cpu' to avoid VRAM issues
device = 'cpu'

# --- Define Available Models ---
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
}

# Dictionary to hold loaded model/tokenizer objects (caching)
LOADED_PIPELINES = {}

# *** Pre-load the smallest model (T5-Small) for quick startup ***
try:
    T5_MODEL_INFO = AVAILABLE_MODELS["T5-Small (Fastest)"]
    T5_TOKENIZER = T5_MODEL_INFO["tokenizer_class"].from_pretrained(T5_MODEL_INFO["path"])
    T5_MODEL = T5_MODEL_INFO["model_class"].from_pretrained(T5_MODEL_INFO["path"])
    T5_MODEL.to(device)

    LOADED_PIPELINES["T5-Small (Fastest)"] = {"model": T5_MODEL, "tokenizer": T5_TOKENIZER}

    print("Pre-loaded T5-Small (Fastest) successfully for quick startup.")
except Exception as e:
    print(f"Failed to pre-load T5-Small: {e}. Relying entirely on lazy loading.")


def get_summarization_pipeline(model_name):
    """Dynamically loads a model/tokenizer if not already in the cache."""
    if model_name in LOADED_PIPELINES:
        return LOADED_PIPELINES[model_name]["model"], LOADED_PIPELINES[model_name]["tokenizer"]

    # If the requested model is not in the cache and not in AVAILABLE_MODELS, raise error
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not recognized or available.")

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


@app.route('/')
def serve_frontend():
    """Serves the main HTML page."""
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.get_json()
    input_text = data.get('text', '')
    min_length = data.get('min_length', 30)
    max_length = data.get('max_length', 150)

    # Defaulting to a safe, pre-loaded model if the frontend sends a bad value
    selected_model_name = data.get('model_name', 'T5-Small (Fastest)')

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    try:
        model, tokenizer = get_summarization_pipeline(selected_model_name)

        # T5 models require a task prefix!
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
    app.run(debug=True, port=5000)