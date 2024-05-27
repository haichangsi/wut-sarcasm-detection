import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pytorch_lightning import Trainer

from bert_headlines_data import *
from utils import *


class HeadlinesSarcasmClassifier(pl.LightningModule):
    def __init__(self, n_classes=2, steps_per_epoch=None, n_epochs=None, lr=2e-5):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(input_ids, attention_mask)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels.cpu(), preds.cpu(), average="weighted"
        )
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_acc", acc, prog_bar=True, logger=True)
        self.log("test_precision", precision, prog_bar=True, logger=True)
        self.log("test_recall", recall, prog_bar=True, logger=True)
        self.log("test_f1", f1, prog_bar=True, logger=True)
        return {
            "test_loss": loss,
            "test_acc": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        total_steps = self.steps_per_epoch * self.n_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        return [optimizer], [scheduler]
