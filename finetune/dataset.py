import pandas as pd
from torch.utils.data import random_split
from transformers import DebertaV2Tokenizer
from models import MODEL_PATH
from torch.utils.data import Dataset, DataLoader
import torch

TRAIN_DATASET_PATH = "./data/train.csv"
TEST_DATASET_PATH = "./data/test.csv"


class APIDocumentDataset(Dataset):
    def __init__(self, data_path = TRAIN_DATASET_PATH):
        df = pd.read_csv(data_path, parse_dates=[])
        # This line is for test on sample data to ensure pipeline is working as expected
        # df = df.iloc[:10]
        
        tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
        def tokenize_text(text):
            return tokenizer(text, padding='max_length', truncation=True, max_length=768, return_tensors='pt')
        
        tokenized_data = df['api_doc'].astype(str).apply(tokenize_text)
        df['input_ids'] = tokenized_data.apply(lambda x: x['input_ids'])
        df['attention_mask'] = tokenized_data.apply(lambda x: x['attention_mask'])
        df['labels'] = df['score'] - 1  # Adjust labels if needed (e.g., to start from 0)
        
        self.df = df
        self.label_count = df.groupby(['labels'])['score'].count().reset_index().to_dict()['score']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        return {
            'input_ids': row['input_ids'],
            'attention_mask': row['attention_mask'],
            'labels': row['labels'],
            'api_doc': row['api_doc'],
        }
        
    @classmethod
    def get_train_dataloaders(cls, train_data_path:str = TRAIN_DATASET_PATH, batch_size=8):
        _dataset = APIDocumentDataset(train_data_path)
        train_size = int(0.8 * len(_dataset))
        val_size = len(_dataset) - train_size
        train_dataset, val_dataset = random_split(_dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, _dataset.label_count
    
    
    @classmethod
    def get_test_dataloaders(cls, test_data_path:str = TEST_DATASET_PATH, batch_size=1):
        _dataset = APIDocumentDataset(test_data_path)
        return DataLoader(_dataset, num_workers=8, batch_size=batch_size, shuffle=False)
        