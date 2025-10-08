import pandas as pd
import json

# Cargar el dataset
with open("train.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Explode para tener un mensaje por fila
df_expanded = df.explode(["messages", "sender_labels", "receiver_labels"])

# Distribución sender_labels
sender_counts = df_expanded["sender_labels"].value_counts()
sender_percent = df_expanded["sender_labels"].value_counts(normalize=True) * 100

sender_distribution = pd.DataFrame({
    "count": sender_counts,
    "percentage": sender_percent
})

print("Distribución sender_labels:")
print(sender_distribution)

# Distribución receiver_labels
receiver_counts = df_expanded["receiver_labels"].value_counts()
receiver_percent = df_expanded["receiver_labels"].value_counts(normalize=True) * 100

receiver_distribution = pd.DataFrame({
    "count": receiver_counts,
    "percentage": receiver_percent
})

print("\nDistribución receiver_labels:")
print(receiver_distribution)
