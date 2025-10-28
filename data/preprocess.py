# improts necesarios
import pandas as pd
import json, re, os
from tqdm import tqdm
import spacy
import emoji
# Para visualizar las barras de progreso
tqdm.pandas()

# Para cargar el modelo de lenguaje en inglés de spaCy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# Función para la limpieza de texto
def clean_text(text):
    # Si no es texto, devuelve una cadena vacía para evitar errores
    if not isinstance(text, str):
        return ""
    # Eliminar URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Eliminar emojis usando la librería emoji
    text = emoji.replace_emoji(text, replace="")
    # Unificar contracciones eliminando comillas simples internas
    # Por ejemplo: don't -> dont, isn't -> isnt
    text = re.sub(r"(\w)'(\w)", r"\1\2", text)
    # Eliminar comillas simples, dobles y otros acentos
    text = re.sub(r"['\"`´]", "", text)
    # Eliminar cualquier carácter no alfabético, manteniendo los espacios
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Pasar a minúsculas
    text = text.lower()
    # Quitar espacios repetidos
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Función para la tokenización y lematización
def preprocess_text(text):
    # Limpieza básica
    text = clean_text(text)
    # Analizamos el texto con spaCy
    doc = nlp(text)
    # Extraemos tokens y lemas, excluyendo stopwords y signos de puntuación
    tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
    lemmas = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
    # Devolvemos un diccionario con el texto limpio, tokens y lemas
    return {"clean": text, "tokens": tokens, "lemmas": lemmas}

# Función para cargar y expandir el dataset
def load_and_expand(path):
    # Leemos linea por línea el archivo JSONL
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Convertimos a DataFrame y explotamos las listas
    df = pd.DataFrame(data)
    return df.explode(["messages", "sender_labels", "receiver_labels"])

def main():
    # Procesar entrenamiento, validación y prueba
    for split in ["train", "validation", "test"]:
        print(f"Procesando {split}...")
        # Cargar y expandir el archivo JSONL
        df = load_and_expand(f"data/{split}.jsonl")
        
        # Aplicar preprocesamiento a cada mensaje con barra de progreso
        df["processed"] = df["messages"].progress_apply(preprocess_text)
          # Extraer las columnas resultantes del diccionario
        df["text_clean"] = df["processed"].apply(lambda d: d["clean"])
        df["tokens"] = df["processed"].apply(lambda d: d["tokens"])
        df["lemmas"] = df["processed"].apply(lambda d: d["lemmas"])

          # Eliminar la columna temporal
        df.drop(columns=["processed"], inplace=True)

        # Eliminar filas cuyo texto limpio esté vacío
        df = df[df["text_clean"].str.strip() != ""]

        # Eliminar filas que contengan emojis, por si alguno persiste
        emoji_pattern = "[" + "".join(emoji.EMOJI_DATA.keys()) + "]"
        df = df[~df["messages"].apply(lambda x: bool(re.search(emoji_pattern, str(x))))]


        # Reiniciar índices después de filtrar
        df.reset_index(drop=True, inplace=True)

        # Asegurar que las columnas de etiquetas sean strings
        for col in ["sender_labels", "receiver_labels"]:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Convertir cualquier columna de tipo object a string para evitar errores al guardar en parquet
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)

        # Guardar a parquet
        out_path = f"data/{split}_preprocessed.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Guardado {out_path}")


if __name__ == "__main__":
    main()