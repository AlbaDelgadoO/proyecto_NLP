# imports necesarios
from transformers import AutoTokenizer, AutoModel
import torch, pandas as pd, numpy as np
from tqdm import tqdm
from pathlib import Path

# Función para extraer embeddings contextuales con BERT
"""
Parámetros:
    - model_name: nombre del modelo BERT preentrenado en Hugging Face
    - split: conjunto de datos a procesar ('train', 'validation', 'test')
    """
def extract_bert_embeddings(model_name="bert-base-uncased", split="train"):
    # Cargamos el tokenizador y el modelo BERT desde Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Colocamos el modelo en modo evaluación: sin gradientes
    model.eval()

    # Vamos a detectar a ver si hay GPU disponible y asignamos dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Cargamos los textos preprocesados
    df = pd.read_parquet(f"data/{split}_preprocessed.parquet")
    texts = df["text_clean"].tolist()

    # Creamos la carpeta de salida para guardar los embeddings
    Path("diplomacy/models/embeddings").mkdir(parents=True, exist_ok=True)
    all_vecs = []

    # Tokenización y extracción de embeddings
    for text in tqdm(texts, desc=f"Extracting {split} embeddings"):
        # Convertimos el texto en tokens compatibles con BERT
        # - truncation=True → corta textos demasiado largos
        # - padding='max_length' → rellena hasta max_length=128
        # - return_tensors='pt' → devuelve tensores de PyTorch
        inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

        # Movemos tensores al dispositivo
        inputs = {k:v.to(device) for k,v in inputs.items()}

        # Desactivamos el cálculo de gradientes
        with torch.no_grad():
            out = model(**inputs)
        cls = out.pooler_output.cpu().numpy()

         # Guardamos el embedding del texto actual
        all_vecs.append(cls.squeeze())

    # Guardamos todos los embeddings en un archivo .npy
    np.save(f"diplomacy/models/embeddings/bert_{split}.npy", np.stack(all_vecs))
    print(f"Embeddings BERT guardados para {split}.")

if __name__ == "__main__":
    extract_bert_embeddings()
