from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoModel
import torch.nn as nn
import torch

class PainterDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=128, returnIDs=False):
        self.returnIDs = returnIDs
        df = pd.read_csv(filename, index_col=0)
        self.IDs = df.index.tolist() if returnIDs else None
        self.texts = df['desc'].tolist()
        self.labels = df['inmuseum'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        tr = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        if self.returnIDs: 
            tr['ID'] = self.IDs[idx]
        return tr

class PainterModel(nn.Module):
    def __init__(self, model_name, n_classes=2, dropout=0.3):
        super(PainterModel, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze embedding layers (optional - comment out to fine-tune)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )        
        last_hidden_states = outputs.last_hidden_state
        cls_token = last_hidden_states[:, 0, :]  # [CLS] token is at position 0
        
        pooled_output = cls_token  # Use the [CLS] token representation
        #pooled_output = torch.mean(last_hidden_states, dim=1)  # Use mean pooling over all tokens TODO: use the mask... 
        # pooled_output = torch.max(last_hidden_states, dim=1).values  # Use max pooling over all tokens TODO: use the mask...

        output = self.dropout(pooled_output)
        output = self.classifier(output)
        
        return output
