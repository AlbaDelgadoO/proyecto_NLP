import pandas as pd
import ast
from gensim.models import Word2Vec, FastText
from pathlib import Path

# Cargar tokens desde parquet
def load_tokens(path):
    """
    Carga la columna 'tokens' desde el parquet y devuelve
    una lista de listas de tokens.
    """
    df = pd.read_parquet(path)
    
    # Si los tokens vienen como string, convertirlos a lista
    if isinstance(df["tokens"].iloc[0], str):
        df["tokens"] = df["tokens"].apply(ast.literal_eval)
    
    # Devolver listas de tokens no nulas
    return df["tokens"].dropna().tolist()

# Entrenamiento Word2Vec
def train_word2vec(train_path="data/train_preprocessed.parquet",
                   output_path="diplomacy/models/embeddings/word2vec.model",
                   vector_size=200, window=5, min_count=5, workers=8, epochs=10):
    """
    Entrena un modelo Word2Vec sobre los tokens del dataset.
    """
    sentences = load_tokens(train_path)

    # Crear carpeta si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Inicializar y entrenar Word2Vec
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,        # Skip-gram
        epochs=epochs
    )

    # Guardar modelo
    model.save(output_path)
    print(f"Word2Vec entrenado y guardado en: {output_path}")

# Entrenamiento FastText
def train_fasttext(train_path="data/train_preprocessed.parquet",
                   output_path="diplomacy/models/embeddings/fasttext.model",
                   vector_size=200, window=5, min_count=3, workers=8, epochs=10):
    """
    Entrena un modelo FastText sobre los tokens del dataset.
    """
    sentences = load_tokens(train_path)

    # Crear carpeta si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Inicializar y entrenar FastText
    model = FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs
    )

    # Guardar modelo
    model.save(output_path)
    print(f"FastText entrenado y guardado en: {output_path}")

if __name__ == "__main__":
    # Entrenar Word2Vec
    train_word2vec()
    # Entrenar FastText
    train_fasttext()
