# imports necesarios
import pandas as pd
import ast
from gensim.models import Word2Vec, FastText
from pathlib import Path

# Cargamos los tokens desde el datset preprocesado
def load_tokens(path):
    # Cargamos el parquet con los mensajes preprocesados
    df = pd.read_parquet(path)
    # Convertimos a listas los tokens si vienen como string
    if isinstance(df["tokens"].iloc[0], str):
        df["tokens"] = df["tokens"].apply(ast.literal_eval)
    # Devolvemos la lista de listas de tokens
    return df["tokens"].dropna().tolist()

# Función para entrenar Word2Vec
def train_word2vec(train_path="data/train_preprocessed.parquet"):
    # Cargamos los tokens del conjunto de entrenamiento
    sentences = load_tokens(train_path)

    # Creamos la carpeta de salida
    Path("diplomacy/models/embeddings").mkdir(parents=True, exist_ok=True)

    # Inicializamos el modelo Word2Vec:
    # - vector_size=200 → dimensión del embedding
    # - window=5 → tamaño de la ventana de contexto
    # - min_count=5 → ignora palabras con frecuencia menor a 5
    # - workers=8 → número de núcleos para paralelizar
    # - sg=1 → usa skip-gram (es mejor para vocabularios pequeños)
    # - epochs=10 → número de iteraciones de entrenamiento
    model = Word2Vec(sentences, vector_size=200, window=5, min_count=5, workers=8, sg=1, epochs=10)
    # Guardamos el modelo entrenado
    model.save("diplomacy/models/embeddings/word2vec.model")
    print("Word2Vec entrenado correctamente.")

# Función para entrenar FastText
def train_fasttext(train_path="data/train_preprocessed.parquet"):
    # Cargamos los tokens del conjunto de entrenamiento
    sentences = load_tokens(train_path)

    # Inicializamos el modelo FastText
    # - min_count=3 → usamos un umbral menor para aprovechar más vocabulario
    model = FastText(sentences, vector_size=200, window=5, min_count=3, workers=8, epochs=10)

    # Guardamos el modelo entrenado
    model.save("diplomacy/models/embeddings/fasttext.model")
    print("FastText entrenado correctamente.")


if __name__ == "__main__":
    train_word2vec()
    train_fasttext()
