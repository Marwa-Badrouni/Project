import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import random
from tqdm import tqdm
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything()

# Load datasets
train = pd.read_parquet('/kaggle/input/unswnb15/UNSW_NB15_training-set.parquet')
test = pd.read_parquet('/kaggle/input/unswnb15/UNSW_NB15_testing-set.parquet')

x_train = train.drop("attack_cat", axis=1)
y_train = train['attack_cat']
x_test = test.drop("attack_cat", axis=1)
y_test = test['attack_cat']
x_train.head()
combined = pd.concat([x_train, x_test], axis=0)

for col in ['proto', 'service', 'state']:
    le = LabelEncoder()
    le.fit(combined[col])
    x_train[col] = le.transform(x_train[col])
    x_test[col] = le.transform(x_test[col])


le_y = LabelEncoder()
y_train = le_y.fit_transform(y_train)
y_test = le_y.transform(y_test)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

indices = np.arange(len(x_train))
np.random.shuffle(indices)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
x_train.shape
def tabular_to_text(X):
    return [ " | ".join([f"{i}: {val:.3f}" for i, val in enumerate(row)]) for row in X ]

X_train_texts = tabular_to_text(x_train)
X_test_texts = tabular_to_text(x_test)
X_train_texts[0]
model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(np.unique(y_train)))

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)
model = get_peft_model(model, lora_config)

# Data parallelism
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)

with torch.no_grad():
    train_tokens = tokenizer(X_train_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    test_tokens = tokenizer(X_test_texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
  class TabularDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = {k: v.clone().detach() for k, v in tokens.items()}
        self.labels = labels.clone().detach()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokens.items()}, self.labels[idx]

train_dataset = TabularDataset(train_tokens, y_train)
test_dataset = TabularDataset(test_tokens, y_test)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

model.train()
for epoch in range(20):
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(train_loader):.4f}")
  
