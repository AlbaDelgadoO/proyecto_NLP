from transformers import AutoTokenizer, AutoModel
import torch, pandas as pd, numpy as np
from tqdm import tqdm
from pathlib import Path

def extract_bert_embeddings(model_name="bert-base-uncased", split="train"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    df = pd.read_parquet(f"data/{split}_preprocessed.parquet")
    texts = df["text_clean"].tolist()

    Path("diplomacy/models/embeddings").mkdir(parents=True, exist_ok=True)
    all_vecs = []

    for text in tqdm(texts, desc=f"Extracting {split} embeddings"):
        inputs = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        cls = out.pooler_output.cpu().numpy()
        all_vecs.append(cls.squeeze())

    np.save(f"diplomacy/models/embeddings/bert_{split}.npy", np.stack(all_vecs))
    print(f"Embeddings BERT guardados para {split}.")

if __name__ == "__main__":
    extract_bert_embeddings()
