import torch
import pandas as pd
from training.metrics import evaluate_metrics
import torch.nn.functional

def train_model(model, train_loader, val_loader, loss_func, optimizer, num_epochs=4, output_file="outputs/training_metrics.csv", dropout_rate=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics_history = []
    dropout = torch.nn.Dropout(dropout_rate)  # Define o dropout

    for epoch in range(num_epochs):
        model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            outputs = dropout(outputs)  # Aplica dropout
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Perda: {loss.item():.4f}")

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_samples

        print(f'Epoch {epoch+1}/{num_epochs} - Perda: {avg_loss:.4f} - Acurácia: {avg_accuracy:.4f}')

        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask)
                
                val_loss += loss_func(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_samples += labels.size(0)

                all_preds.extend(outputs.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / val_samples
        avg_val_accuracy = val_correct / val_samples

        print(f'Acurácia de Validação: {avg_val_accuracy:.4f} - Perda de Validação: {avg_val_loss:.4f}')

        val_metrics = evaluate_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
        print(f'Métricas de Validação: {val_metrics}')

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_accuracy": avg_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_accuracy,
            "f1_score": val_metrics["f1_score"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"]
        })

        df = pd.DataFrame(metrics_history)
        df.to_csv(output_file, index=False)

    print(f"✅ Treinamento concluído! Métricas salvas em {output_file}")