import torch
from torch import nn
from transformers import DebertaV2Model


MODEL_PATH = "/Users/jiehu/Documents/LLM/deberta-v3-base"

class CustomDebertaClassifier(nn.Module):
    def __init__(self, num_labels=5, num_layers=1, device='mps'):
        super(CustomDebertaClassifier, self).__init__()
        self.device = torch.device(device)

        # Load and freeze DeBERTa
        self.deberta = DebertaV2Model.from_pretrained(MODEL_PATH)
        # freeze weights
        for param in self.deberta.parameters():
            param.requires_grad = False

        # Additional layers
        deberta_hidden_size = self.deberta.config.hidden_size
        layers = [nn.Linear(deberta_hidden_size, deberta_hidden_size) for _ in range(num_layers)]
        self.additional_layers = nn.Sequential(*layers)

        # Final classifier layer
        self.classifier = nn.Linear(deberta_hidden_size, num_labels)

        self.to(self.device)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1))
        
        pooled_output = outputs.last_hidden_state[:, 0, :]

        for layer in self.additional_layers:
            pooled_output = layer(pooled_output)

        return self.classifier(pooled_output)

    