import pandas as pd
import torch
from transformers import BertTokenizer, RobertaTokenizer, DebertaTokenizerFast
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

def tokenize_texts(texts, model_type='bert', max_length=128):
    
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    else:
        raise ValueError("Erro: Modelo não reconhecido")

    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

def load_data(file_path="data/ecommerceDataset.csv", val_split=0.2, batch_size=16, model_type='bert', fraction=0.2):

    df = pd.read_csv(file_path, header=None)
    df.columns = ['Classification', 'Text']

    df = df.sample(frac=fraction, random_state=42).reset_index(drop=True)

    categories = df["Classification"].unique()
    category_to_number = {category: idx for idx, category in enumerate(categories)}
    df["Class_num"] = df["Classification"].map(category_to_number)

    df = df.dropna(subset=['Classification', 'Text', "Class_num"])

    tokenized = tokenize_texts(df['Text'].tolist(), model_type=model_type)

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = torch.tensor(df["Class_num"].values)

    dataset = TensorDataset(input_ids, attention_mask, labels)

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(categories)


def plot_loss(csv_path="outputs/training_metrics.csv"):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Treinamento", marker="o", linestyle="-")
    plt.plot(df["epoch"], df["val_loss"], label="Validação", marker="s", linestyle="--")
    plt.title("Evolução da Loss")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

def plot_accuracy(csv_path="outputs/training_metrics.csv"):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_accuracy"], label="Treinamento", marker="o", linestyle="-")
    plt.plot(df["epoch"], df["val_accuracy"], label="Validação", marker="s", linestyle="--")
    plt.title("Evolução da Acurácia")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid()
    plt.show()

def plot_f1_metrics(csv_path="outputs/training_metrics.csv"):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["precision"], label="Precisão", marker="o", linestyle="-")
    plt.plot(df["epoch"], df["recall"], label="Recall", marker="s", linestyle="--")
    plt.plot(df["epoch"], df["f1_score"], label="F1-Score", marker="D", linestyle="-.")
    plt.title("Métricas de Classificação")
    plt.xlabel("Épocas")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid()
    plt.show()