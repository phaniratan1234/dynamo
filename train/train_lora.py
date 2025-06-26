import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Config
TASK_NAME = "sst2"
TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "train_glue_sst2_20250622_192827.json")
VAL_PATH = os.path.join(BASE_DIR, "data", "raw", "validation_glue_sst2_20250622_192827.json")
SAVE_DIR = f"../models/saved_adapters/{TASK_NAME}/"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SST2Dataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=128):
        with open(json_path, "r") as f:
            self.samples = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        encoding = self.tokenizer(
            item["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"])
        }

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    return accuracy_score(labels, preds)

def main():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = SST2Dataset(TRAIN_PATH, tokenizer)
    val_dataset = SST2Dataset(VAL_PATH, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    base_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    for param in base_model.roberta.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(base_model, lora_config)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train(model, train_loader, optimizer)
        acc = evaluate(model, val_loader)
        print(f"Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f}")

    model.save_pretrained(SAVE_DIR)
    torch.save(model.state_dict(), "lora_adapter_sst2.pth")
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ Model saved to {SAVE_DIR}")

if __name__ == '__main__':
    main()
