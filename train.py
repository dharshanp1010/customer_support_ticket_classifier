import torch
from transformers import AutoModelForSequenceClassification
from data_preprocessing import train_loader, val_loader, model_name, encoder
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(encoder.classes_))
device = torch.device("cpu")
model = model.to(device)

input_ids, attention_mask, labels = next(iter(train_loader))

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 2

for epoch in range(epochs):

    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for input_ids, attention_mask, labels in tqdm(train_loader, desc="Training"):

        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
        logits = outputs.logits
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0) #mean per batch
        predictions = logits.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(val_loader, desc="Validation"):

            labels = labels.to(device)
            outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            logits = outputs.logits
            loss = criterion(logits, labels)

            val_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    train_accuracy = 100 * train_correct / train_total
    val_accuracy = 100 * val_correct / val_total

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Train Loss: {train_loss / train_total:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Val Loss: {val_loss / val_total:.4f}")
    print(f"Val Accuracy: {val_accuracy:.2f}%")

# Saving trained model
model.config.id2label = dict(enumerate(encoder.classes_))
model.config.label2id = {
    category: i for i, category in enumerate(encoder.classes_)
}

model.save_pretrained("trained_model")