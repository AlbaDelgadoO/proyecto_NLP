from gensim.models import Word2Vec, FastText
import pandas as pd
from pathlib import Path

def train_word2vec(train_path="data/train_preprocessed.parquet"):
    df = pd.read_parquet(train_path)
    sentences = df["tokens"].dropna().tolist()

    Path("diplomacy/models/embeddings").mkdir(parents=True, exist_ok=True)

    model = Word2Vec(sentences, vector_size=200, window=5, min_count=5, workers=8, sg=1, epochs=10)
    model.save("diplomacy/models/embeddings/word2vec.model")
    print("Word2Vec entrenado y guardado.")

def train_fasttext(train_path="data/train_preprocessed.parquet"):
    df = pd.read_parquet(train_path)
    sentences = df["tokens"].dropna().tolist()

    model = FastText(sentences, vector_size=200, window=5, min_count=3, workers=8, epochs=10)
    model.save("diplomacy/models/embeddings/fasttext.model")
    print("FastText entrenado y guardado.")

if __name__ == "__main__":
    train_word2vec()
    train_fasttext()
