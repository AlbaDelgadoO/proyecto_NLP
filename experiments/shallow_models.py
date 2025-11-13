import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import joblib

# Cargar datos y etiquetas
X_tfidf = load_npz("../diplomacy/models/representations/X_tfidf_train.npz")
X_bow = load_npz("../diplomacy/models/representations/X_bow_train.npz")
df = pd.read_parquet("../data/train_preprocessed.parquet")
"""
# Convertir las etiquetas a enteros (0 = False, 1 = True) porque en train_preprocessed.parquet la columna 
sender_labels está guardada como string ('True' / 'False') en lugar de valores booleanos (True / False).
Por eso, al hacer .astype(int), Python intenta convertir literalmente el texto 'True' en número y falla.
"""
y = df["sender_labels"].astype(str).str.lower().map({"true": 1, "false": 0})

# Ver distribución de clases
print("Distribución de clases en sender_labels:")
print(y.value_counts(normalize=True).rename("porcentaje (%)") * 100)
print("\nConteo absoluto:")
print(y.value_counts())
print("-" * 50)

# División train/val
X_train_tfidf, X_val_tfidf, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
X_train_bow, X_val_bow, _, _ = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)

# OVERSAMPLING (balancear clases)
ros = RandomOverSampler(random_state=42)
X_train_tfidf_bal, y_train_bal = ros.fit_resample(X_train_tfidf, y_train)
X_train_bow_bal, _ = ros.fit_resample(X_train_bow, y_train)
print("Tamaño tras oversampling:", X_train_tfidf_bal.shape)
print("Distribución balanceada:")
print(pd.Series(y_train_bal).value_counts())
print("-" * 50)

# Ambos modelos van a usar class_weight="balanced" para compensar el desbalance de clases.
# REGRESIÓN LOGÍSTICA con TF-IDF
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_tfidf_bal, y_train_bal)

# Ajuste de umbral (probabilidad >= 0.4)
y_probs = lr.predict_proba(X_val_tfidf)[:, 1]
y_pred_lr = (y_probs >= 0.4).astype(int)

print("Logistic Regression (TF-IDF, oversampling, threshold=0.4)")
print(classification_report(y_val, y_pred_lr))
joblib.dump(lr, "../diplomacy/models/shallow/logreg_tfidf.joblib")

# SVM LINEAL con BoW
svm = LinearSVC(max_iter=2000)
svm.fit(X_train_bow_bal, y_train_bal)
y_pred_svm = svm.predict(X_val_bow)
print("Linear SVM (BoW, oversampling)")
print(classification_report(y_val, y_pred_svm))
joblib.dump(svm, "../diplomacy/models/shallow/svm_bow.joblib")

# XGBOOST (TF-IDF)
xgb = XGBClassifier(
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # balanceo interno
    eval_metric="logloss",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
)
xgb.fit(X_train_tfidf, y_train)
y_pred_xgb = xgb.predict(X_val_tfidf)
print("XGBoost (TF-IDF)")
print(classification_report(y_val, y_pred_xgb))
joblib.dump(xgb, "../diplomacy/models/shallow/xgb_tfidf.joblib")