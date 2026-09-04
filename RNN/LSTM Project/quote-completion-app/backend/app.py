"""
Backend API for the Automatic Sentence Completion project.

Loads the trained LSTM (model.h5) plus the Tokenizer that was fit on the
training corpus (tokenizer.pickle), and exposes a small HTTP API the
frontend (frontend/index.html) talks to.

IMPORTANT — tokenizer.pickle is required and is NOT included:
The notebook never saved the Keras Tokenizer it fit on qoute_dataset.csv,
only the model weights (model.h5). The model's predictions only make sense
if words are converted to the exact same integer ids it was trained on, so
this file cannot be guessed or rebuilt from the model alone.

To produce it, run generate_tokenizer.py in this folder against the
original qoute_dataset.csv (see README.md for the one-line notebook
addition that saves it directly, which is the more reliable option).
"""

import os
import pickle

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.h5")
TOKENIZER_PATH = os.path.join(APP_DIR, "tokenizer.pickle")

MAX_LEN = 745  # fixed input length the model was trained on (from model_config)

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None
index_to_word = {}
load_error = None

try:
    model = load_model(MODEL_PATH)
except Exception as exc:  # pragma: no cover
    load_error = f"Could not load model.h5: {exc}"

if os.path.exists(TOKENIZER_PATH):
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    except Exception as exc:  # pragma: no cover
        load_error = f"Could not load tokenizer.pickle: {exc}"
else:
    load_error = (
        "tokenizer.pickle is missing from backend/. The app cannot turn words "
        "into the model's vocabulary ids without it — see README.md."
    )


def predict_next_word(seed_text: str) -> str:
    seq = tokenizer.texts_to_sequences([seed_text.lower()])[0]
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding="pre")
    pred = model.predict(padded, verbose=0)
    pred_index = int(np.argmax(pred))
    return index_to_word.get(pred_index, "")


def generate_words(seed_text: str, n_words: int) -> str:
    text = seed_text
    for _ in range(n_words):
        next_word = predict_next_word(text)
        if not next_word:
            break
        text += " " + next_word
    return text


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ready": model is not None and tokenizer is not None,
            "error": load_error,
        }
    )


@app.post("/api/predict")
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": load_error or "Model not ready."}), 503

    data = request.get_json(silent=True) or {}
    seed_text = (data.get("text") or "").strip()
    n_words = int(data.get("n_words", 5))
    n_words = max(1, min(n_words, 20))

    if not seed_text:
        return jsonify({"error": "Please provide some starting text."}), 400

    completed = generate_words(seed_text, n_words)
    added = completed[len(seed_text):].strip()

    return jsonify({"seed": seed_text, "completion": added, "full_text": completed})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
