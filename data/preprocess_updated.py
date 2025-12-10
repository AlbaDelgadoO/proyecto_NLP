import pandas as pd
import json, re, os
from tqdm import tqdm
import spacy
import emoji

# Para visualizar las barras de progreso
tqdm.pandas()

# Cargar spaCy
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    print("Descargando modelo spaCy...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

# ---------------------------------------------------------
# 1. Funciones de Limpieza
# ---------------------------------------------------------

def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"(\w)'(\w)", r"\1\2", text)
    text = re.sub(r"['\"`´]", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text) 
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Limpieza suave - ideal para Transformers (RoBERTa/BERT)
def clean_text_minimal(text):
    if not isinstance(text, str): return ""
    # Eliminamos URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Eliminamos emojis
    text = emoji.replace_emoji(text, replace="")
    # Quitamos espacios repetidos y espacios al inicio/final
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------------------------------------
# 2. Procesamiento de Texto
# ---------------------------------------------------------
def preprocess_text(text):
    clean_agg = clean_text(text)
    
    doc = nlp(clean_agg)
    tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
    lemmas = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
    
    clean_min = clean_text_minimal(text)
    
    return {
        "clean_aggressive": clean_agg, 
        "clean_minimal": clean_min,
        "tokens": tokens, 
        "lemmas": lemmas
    }

# ---------------------------------------------------------
# 3. Carga y Expansión 
# ---------------------------------------------------------
def load_and_expand(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    
    # Buscamos las columnas de hablantes y receptores para "explotarlas"
    cols_to_explode = ["messages", "sender_labels", "receiver_labels", "speakers", "receivers"]
    
    # Filtramos solo las columnas que realmente existen en el JSON
    cols_to_explode = [c for c in df.columns if c in cols_to_explode]
    
    # Explotamos: convierte listas en filas individuales
    df = df.explode(cols_to_explode)
    return df

def main():
    for split in ["train", "validation", "test"]:
        input_path = f"{split}.jsonl"
        
        if not os.path.exists(input_path):
            print(f"Archivo no encontrado: {input_path}. Saltando...")
            continue
            
        print(f"Procesando {split}...")
        
        # 1. Cargar y expandir
        df = load_and_expand(input_path)

        # 2. Preprocesamiento de texto
        print("  - Limpiando textos y generando tokens...")
        df["processed"] = df["messages"].progress_apply(preprocess_text)
        
        # Desempaquetar resultados
        df["text_clean"] = df["processed"].apply(lambda d: d["clean_aggressive"]) 
        df["msg_for_context"] = df["processed"].apply(lambda d: d["clean_minimal"])
        df["tokens"] = df["processed"].apply(lambda d: d["tokens"])
        df["lemmas"] = df["processed"].apply(lambda d: d["lemmas"])
        
        df.drop(columns=["processed"], inplace=True)

        # 3. Filtrado de vacíos
        df = df[df["text_clean"].str.strip() != ""]
        emoji_pattern = "[" + "".join(emoji.EMOJI_DATA.keys()) + "]"
        df = df[~df["messages"].apply(lambda x: bool(re.search(emoji_pattern, str(x))))]
        df.reset_index(drop=True, inplace=True)

        # 4. Convertir columnas a string
        if "speakers" in df.columns:
            df["speakers"] = df["speakers"].fillna("Unknown").astype(str)
        if "receivers" in df.columns:
            df["receivers"] = df["receivers"].fillna("Unknown").astype(str)
            
        for col in ["sender_labels", "receiver_labels"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # ---------------------------------------------------------
        # 5. INGENIERÍA DE CONTEXTO
        # ---------------------------------------------------------
        print("  - Creando columna de contexto...")
        
        # Formato: "Speaker -> Receiver: Mensaje"
        if "speakers" in df.columns and "receivers" in df.columns:
            df["text_context"] = (
                df["speakers"] + " -> " + df["receivers"] + ": " + df["msg_for_context"]
            )
        else:
            print("  ADVERTENCIA: No se encontraron columnas 'speakers'/'receivers'. Usando solo mensaje.")
            df["text_context"] = df["msg_for_context"]

        # Limpieza final de tipos object
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)

        # 6. Guardar con NUEVO NOMBRE
        out_path = f"{split}_with_context.parquet" # <--- AQUI CAMBIÓ EL NOMBRE
        df.to_parquet(out_path, index=False)
        print(f"Guardado: {out_path} ({len(df)} filas)")

if __name__ == "__main__":
    main()