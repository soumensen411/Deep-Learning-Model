# The Commonplace Book — frontend for the sentence-completion model

A small app around `model.h5`: type the start of a sentence, and the
LSTM predicts how it continues, one word at a time.

```
quote-completion-app/
├── backend/
│   ├── app.py                 Flask API (/api/predict, /api/health)
│   ├── model.h5                your trained model
│   ├── generate_tokenizer.py   rebuilds tokenizer.pickle from the CSV
│   └── requirements.txt
├── frontend/
│   └── index.html              the UI (open directly in a browser)
└── README.md
```

## One missing piece: tokenizer.pickle

The notebook fits a `Tokenizer` on `qoute_dataset.csv` and uses it to turn
words into the integer ids the model was trained on — but the notebook only
ever saves `model.h5`, never the tokenizer itself. Without that exact
word → id mapping, predictions would be meaningless, so `backend/app.py`
refuses to run inference until `tokenizer.pickle` exists next to it.

Two ways to get it, pick whichever is easier:

**Option A — add one cell to the notebook** (most reliable, guaranteed to
match the trained weights exactly). Right after `tokenizer.fit_on_texts(df)`,
add:

```python
import pickle
with open('tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)
```

Re-run that cell, download `tokenizer.pickle`, and drop it into `backend/`.

**Option B — run the included script** against the original
`qoute_dataset.csv` (the file the notebook reads from Google Drive):

```bash
cd backend
python generate_tokenizer.py /path/to/qoute_dataset.csv
```

This repeats the notebook's exact preprocessing (lowercase → strip
punctuation → `Tokenizer(num_words=8980)`) so the resulting vocabulary
lines up with the trained model.

## Running it

```bash
cd backend
pip install -r requirements.txt
python app.py
```

This starts the API at `http://localhost:5000`. Then just open
`frontend/index.html` in a browser — no build step needed. The footer at
the bottom of the page shows whether the model loaded successfully.

## API

`POST /api/predict`
```json
{ "text": "the meaning of life is", "n_words": 5 }
```
→
```json
{ "seed": "the meaning of life is", "completion": "not", "full_text": "the meaning of life is not" }
```

`GET /api/health` — reports whether the model and tokenizer are loaded.
