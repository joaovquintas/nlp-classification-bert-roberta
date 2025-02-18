import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from training.metrics import evaluate_metrics 
import numpy as np

def train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs=4):
    device = torch.device("cpu")  # Se você tiver GPU, use 'cuda'
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0
        total_accuracy = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += (outputs.argmax(1) == labels).sum().item()

            if batch_idx % 10 == 0:  # Mostra progresso a cada 10 batches
                print(f"🟢 Lote {batch_idx}/{len(train_loader)} - Perda: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_accuracy = total_accuracy / len(train_loader.dataset)
        
        print(f'Época {epoch+1}/{num_epochs} - Perda: {avg_loss:.4f} - Acurácia: {avg_accuracy:.4f}')
        
        # Avaliação nas métricas de validação
        model.eval()
        all_preds = []  
        all_labels = []  

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask)

                # Obter a classe com a maior probabilidade
                pred_classes = torch.argmax(outputs, dim=1)

                all_preds.append(pred_classes.cpu().numpy())  # Armazenar predições
                all_labels.append(labels.cpu().numpy())       # Armazenar os rótulos reais


        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        val_metrics = evaluate_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
        print(f'Métricas de Validação: {val_metrics}')
