import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from data_preprocessing import test_loader, encoder

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained("trained_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

actual_labels = []
predicted_labels = []

# Testing
with torch.no_grad():

    for input_ids, attention_mask, labels in tqdm(test_loader, desc="Testing"):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        predictions = outputs.logits.argmax(dim=1)

        actual_labels.extend(labels.tolist())
        predicted_labels.extend(predictions.cpu().tolist())


print("Test Accuracy:", accuracy_score(actual_labels, predicted_labels) * 100)

print(classification_report(actual_labels,predicted_labels,target_names=encoder.classes_,zero_division=0))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(actual_labels,predicted_labels,display_labels=encoder.classes_,xticks_rotation=45,cmap="Blues")

plt.tight_layout()
plt.show()