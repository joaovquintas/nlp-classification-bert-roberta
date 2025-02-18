import torch.nn as nn
from transformers import RobertaModel

class RobertaClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')

        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        pool_output = outputs.pooler_output

        logits = self.fc(pool_output)

        return logits