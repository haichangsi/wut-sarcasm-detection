from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pytorch_lightning as pl
import torch

from utils import *


class HeadlinesSarcasmDataset(Dataset):
    def __init__(self, data, max_token_len=256):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["headline"]
        label = self.data.iloc[index]["is_sarcastic"]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class HeadlinesSarcasmDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=16):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = HeadlinesSarcasmDataset(self.train_data)
        self.val_dataset = HeadlinesSarcasmDataset(self.val_data)
        self.test_dataset = HeadlinesSarcasmDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
