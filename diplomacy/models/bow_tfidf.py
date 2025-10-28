# imports necesarios
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib, scipy.sparse as sp
from pathlib import Path

# Función para generar representaciones BOW y TF-IDF
def build_bow_tfidf(train_path="data/train_preprocessed.parquet"):
    # Cargamos el conjunto de entrenamiento preprocesado
    df = pd.read_parquet(train_path)
    # Obtenemos los textos limpios: columna generada en preprocess.py
    texts = df["text_clean"].fillna("").tolist()

     # Creamos la carpeta de salida para guardar los modelos y matrices
    Path("diplomacy/models/representations").mkdir(parents=True, exist_ok=True)

    # Bag-of-Words:
    # Inicializamos el vectorizador BOW
    # - ngram_range=(1,2): considera unigramas y bigramas
    # - min_df=5: ignora términos que aparecen en menos de 5 documentos
    # - max_features=30000: limita el tamaño máximo del vocabulario
    bow = CountVectorizer(ngram_range=(1,2), min_df=5, max_features=30000)

    # Ajustamos el modelo y transformar los textos en una matriz dispersa
    X_bow = bow.fit_transform(texts)
     # Guardamos el vectorizador entrenado con joblib, para usarlo en validación o test
    joblib.dump(bow, "diplomacy/models/representations/bow_vectorizer.joblib")
    # Guardamos la matriz de características en formato .npz porque es eficiente para matrices grandes y dispersas
    sp.save_npz("diplomacy/models/representations/X_bow_train.npz", X_bow)

    # Hacemos print del tamaño del vocabulario
    print("BOW listo. Vocab size:", len(bow.vocabulary_))

    # TF-IDF:
    # Inicializamos el vectorizador TF-IDF con los mismos parámetros que BOW
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=30000)

    # Ajustamos el modelo y transformar los textos en una matriz dispersa
    X_tfidf = tfidf.fit_transform(texts)
    # Guardamos el vectorizador TF-IDF y su matriz
    joblib.dump(tfidf, "diplomacy/models/representations/tfidf_vectorizer.joblib")
    sp.save_npz("diplomacy/models/representations/X_tfidf_train.npz", X_tfidf)

    # Hacemos print del tamaño del vocabulario
    print("TF-IDF listo. Vocab size:", len(tfidf.vocabulary_))

if __name__ == "__main__":
    build_bow_tfidf()