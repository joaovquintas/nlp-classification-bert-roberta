import torch.nn as nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pool_output = outputs.pooler_output #CLS

        logits = self.fc(pool_output)

        return logits
