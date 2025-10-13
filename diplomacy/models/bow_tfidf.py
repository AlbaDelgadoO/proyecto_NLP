import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib, scipy.sparse as sp
from pathlib import Path

def build_bow_tfidf(train_path="data/train_preprocessed.parquet"):
    df = pd.read_parquet(train_path)
    texts = df["text_clean"].fillna("").tolist()

    Path("diplomacy/models/representations").mkdir(parents=True, exist_ok=True)

    # Bag-of-Words
    bow = CountVectorizer(ngram_range=(1,2), min_df=5, max_features=30000)
    X_bow = bow.fit_transform(texts)
    joblib.dump(bow, "diplomacy/models/representations/bow_vectorizer.joblib")
    sp.save_npz("diplomacy/models/representations/X_bow_train.npz", X_bow)
    print("BOW listo. Vocab size:", len(bow.vocabulary_))

    # TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=30000)
    X_tfidf = tfidf.fit_transform(texts)
    joblib.dump(tfidf, "diplomacy/models/representations/tfidf_vectorizer.joblib")
    sp.save_npz("diplomacy/models/representations/X_tfidf_train.npz", X_tfidf)
    print("TF-IDF listo. Vocab size:", len(tfidf.vocabulary_))

if __name__ == "__main__":
    build_bow_tfidf()