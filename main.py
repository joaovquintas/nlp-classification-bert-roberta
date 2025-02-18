import torch
import torch.optim as optim
from torch import nn
from models.bert_classifier import BertClassifier
from models.roberta_classifier import RobertaClassifier
from training.train import train_model
from models.utils import load_data

def get_model(model_type='bert', num_classes=10):
    """
    Retorna o modelo de classificação baseado no tipo de modelo escolhido.

    Args:
        model_type (str): 'bert' ou 'roberta'
        num_classes (int): Número de classes no dataset

    Returns:
        Modelo instanciado (BertClassifier ou RobertaClassifier)
    """
    if model_type == 'bert':
        model = BertClassifier(num_classes=num_classes)
    elif model_type == 'roberta':
        model = RobertaClassifier(num_classes=num_classes)
    else:
        raise ValueError("Erro: Modelo não reconhecido")

    return model
 
def main():
    print("🚀 Iniciando script...")

    model_type = 'roberta'
    print(f"🔍 Modelo escolhido: {model_type}")

    print("📂 Carregando dataset...")
    train_loader, val_loader, num_classes = load_data("data/ecommerceDataset.csv", model_type=model_type, fraction=0.01)
    print(f"✅ Dataset carregado com {num_classes} classes!")

    print("🛠️ Criando modelo...")
    model = get_model(model_type=model_type, num_classes=num_classes)
    print("✅ Modelo instanciado!")

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-5) #2e-5

    print("🎯 Iniciando treinamento...")
    train_model(model, train_loader, val_loader, loss_func, optimizer)
    print("🏁 Treinamento finalizado!")

if __name__ == "__main__":
    main()
