import pytorch_lightning as pl
from argparse import Namespace
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    AdamW,
    DataCollatorForLanguageModeling
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

args = Namespace()
args.train = "data/train.txt"
args.max_len = 128
args.model_name = "bert-base-uncased"
args.epochs = 1
args.batch_size = 4

tokenizer = BertTokenizer.from_pretrained(args.model_name)

class MaskedLMDataset(Dataset):
    def __init__(self, file, tokenizer):
        self.tokenizer = tokenizer
        self.lines = self.load_lines(file)
        self.ids = self.encode_lines(self.lines)

    def load_lines(self, file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines

    def encode_lines(self, lines):
        batch_encoding = self.tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=args.max_len
        )
        return batch_encoding["input_ids"]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)

class Bert(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(args.model_name)

    def forward(self, input_ids, labels):
        return self.bert(input_ids=input_ids,labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = self(input_ids=input_ids, labels=labels)
        loss = outputs[0]
        return {"loss": loss}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)


def train():
    train_dataset = MaskedLMDataset(args.train, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator
    )
    model = Bert()

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1)
    trainer.fit(model, train_loader)
    torch.save(model.state_dict(), 'checkpoints/fine_tune_mlm.bin')


class BertPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, labels=None):
        return self.bert(input_ids=input_ids, labels=labels)

def infer():
    new_model = BertPred()
    new_model.load_state_dict(torch.load('checkpoints/fine_tune_mlm.bin'))
    new_model.eval()


if __name__ == "__main__":
    train()
