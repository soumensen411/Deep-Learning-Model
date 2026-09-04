"""
Regenerates tokenizer.pickle from the original qoute_dataset.csv, using the
exact same preprocessing steps as Automatic_sentence_completion.ipynb, so the
word -> id mapping matches what model.h5 was trained on.

Usage:
    python generate_tokenizer.py /path/to/qoute_dataset.csv

Then copy the resulting tokenizer.pickle into this backend/ folder
(it's written here directly by default).
"""

import pickle
import string
import sys

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

VOCAB_SIZE = 8980  # must match the notebook / model_config exactly


def main(csv_path: str, out_path: str = "tokenizer.pickle"):
    df = pd.read_csv(csv_path)
    quotes = df["quote"].str.lower()

    translator = str.maketrans("", "", string.punctuation)
    quotes = quotes.apply(lambda x: x.translate(translator))

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(quotes)

    with open(out_path, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"Saved {out_path} ({len(tokenizer.word_index)} words in vocabulary)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_tokenizer.py /path/to/qoute_dataset.csv")
        sys.exit(1)
    main(sys.argv[1])
