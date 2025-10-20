import pandas as pd
import ast
from gensim.models import Word2Vec, FastText
from pathlib import Path

def load_tokens(path):
    df = pd.read_parquet(path)
    # Convertir a listas si vienen como string
    if isinstance(df["tokens"].iloc[0], str):
        df["tokens"] = df["tokens"].apply(ast.literal_eval)
    return df["tokens"].dropna().tolist()

def train_word2vec(train_path="data/train_preprocessed.parquet"):
    sentences = load_tokens(train_path)

    Path("diplomacy/models/embeddings").mkdir(parents=True, exist_ok=True)
    model = Word2Vec(sentences, vector_size=200, window=5, min_count=5, workers=8, sg=1, epochs=10)
    model.save("diplomacy/models/embeddings/word2vec.model")
    print("Word2Vec entrenado correctamente.")

def train_fasttext(train_path="data/train_preprocessed.parquet"):
    sentences = load_tokens(train_path)
    model = FastText(sentences, vector_size=200, window=5, min_count=3, workers=8, epochs=10)
    model.save("diplomacy/models/embeddings/fasttext.model")
    print("FastText entrenado correctamente.")


if __name__ == "__main__":
    train_word2vec()
    train_fasttext()
