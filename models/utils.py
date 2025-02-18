import pandas as pd
import torch
from transformers import BertTokenizer, RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset, random_split

def tokenize_texts(texts, model_type='bert', max_length=128):
    """
    Tokeniza os textos usando BERT ou RoBERTa.
    """
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

    # Reduz o dataset para a fração desejada
    df = df.sample(frac=fraction, random_state=42).reset_index(drop=True)

    # Mapeia as categorias para números
    categories = df["Classification"].unique()
    category_to_number = {category: idx for idx, category in enumerate(categories)}
    df["Class_num"] = df["Classification"].map(category_to_number)

    # Remove valores ausentes
    df = df.dropna(subset=['Classification', 'Text', "Class_num"])

    # Tokeniza os textos com o modelo correto
    tokenized = tokenize_texts(df['Text'].tolist(), model_type=model_type)

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = torch.tensor(df["Class_num"].values)

    # Cria dataset
    dataset = TensorDataset(input_ids, attention_mask, labels)

    # Divide em treino e validação
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Converte para DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(categories)
