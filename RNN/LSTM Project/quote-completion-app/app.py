"""
Automatic Sentence Completion — Flask backend

Serves a small web UI that talks to the trained LSTM next-word
prediction model (model.h5 + tokenizer.pkl + max_len.pkl).
"""

import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

MODEL_DIR = "model"

# ---------------------------------------------------------------------------
# Load model + tokenizer + max_len once at startup
# ---------------------------------------------------------------------------
print("Loading model...")
model = load_model(f"{MODEL_DIR}/model.h5")

print("Loading tokenizer...")
with open(f"{MODEL_DIR}/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("Loading max_len...")
with open(f"{MODEL_DIR}/max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Build index -> word lookup once (same as the notebook)
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

print(f"Ready. vocab_size={tokenizer.num_words}, max_len={max_len}")


def predict_next_word(text: str) -> str:
    """Predict a single next word for the given seed text."""
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")

    pred = model.predict(seq, verbose=0)
    pred_index = int(np.argmax(pred))
    return index_to_word.get(pred_index, "")


def predict_top_k(text: str, k: int = 5):
    """Return the top-k candidate next words with their probabilities."""
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")

    pred = model.predict(seq, verbose=0)[0]
    top_indices = pred.argsort()[-k:][::-1]

    results = []
    for idx in top_indices:
        word = index_to_word.get(int(idx), "")
        if word:
            results.append({"word": word, "probability": float(pred[idx])})
    return results


def generate_text(seed_text: str, n_words: int) -> str:
    """Repeatedly predict and append the next word, n_words times."""
    text = seed_text
    for _ in range(n_words):
        next_word = predict_next_word(text)
        if not next_word:
            break
        text += " " + next_word
    return text


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Return the top-k next-word suggestions for the given text."""
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    k = int(data.get("k", 5))

    if not text:
        return jsonify({"error": "text is required"}), 400

    suggestions = predict_top_k(text, k=k)
    return jsonify({"suggestions": suggestions})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Autocomplete the given seed text with n_words additional words."""
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    n_words = int(data.get("n_words", 5))
    n_words = max(1, min(n_words, 30))  # keep requests reasonable

    if not text:
        return jsonify({"error": "text is required"}), 400

    completed = generate_text(text, n_words)
    return jsonify({"completed": completed})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)