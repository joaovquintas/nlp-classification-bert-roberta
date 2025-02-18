# training/metrics.py
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def accuracy(preds, labels):
    correct = (preds == labels).float()
    return correct.sum() / len(correct)

def evaluate_metrics(preds, labels):
    """
    Função para calcular as métricas de avaliação (Acurácia, F1, Precisão, Recall).
    """
    preds = preds.cpu().numpy()  # As predições já são classes, não são logits
    labels = labels.cpu().numpy()

    accuracy_value = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')

    return {"accuracy": accuracy_value, "f1_score": f1, "precision": precision, "recall": recall}

def save_metrics(model_name, metrics):

    df = pd.DataFrame([metrics])
    df.to_csv(f'outputs/{model_name}_metrics.csv', mode='a', header=False, index=False)