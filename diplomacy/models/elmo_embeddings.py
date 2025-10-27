from allennlp.commands.elmo import ElmoEmbedder
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch

def extract_elmo_embeddings(split="train"):
    # Inicializar el modelo ELMo preentrenado
    elmo = ElmoEmbedder(
        options_file="https://allennlp.s3.amazonaws.com/elmo/2x4096_512_2048cnn_2xhighway/options.json",
        weight_file="https://allennlp.s3.amazonaws.com/elmo/2x4096_512_2048cnn_2xhighway/elmo_weights.hdf5",
        cuda_device=0 if torch.cuda.is_available() else -1
    )

    # Cargar el dataset preprocesado
    df = pd.read_parquet(f"data/{split}_preprocessed.parquet")
    texts = df["text_clean"].tolist()

    Path("diplomacy/models/embeddings").mkdir(parents=True, exist_ok=True)

    all_vecs = []

    for text in tqdm(texts, desc=f"Extracting ELMo embeddings for {split}"):
        tokens = text.split()
        if not tokens:
            all_vecs.append(np.zeros(1024))
            continue
        # ELMo devuelve una tupla (layer1, layer2, layer3)
        # Tomamos la media de todas las capas y de todos los tokens
        vecs = elmo.embed_sentence(tokens)
        mean_vec = vecs.mean(axis=(0, 1))  # promedio por capa y token
        all_vecs.append(mean_vec)

    all_vecs = np.stack(all_vecs)
    np.save(f"diplomacy/models/embeddings/elmo_{split}.npy", all_vecs)
    print(f"Embeddings ELMo guardados para {split} (shape={all_vecs.shape}).")

if __name__ == "__main__":
    extract_elmo_embeddings()
