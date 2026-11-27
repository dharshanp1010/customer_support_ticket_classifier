import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load tokenizer and trained model
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("trained_model")

model.eval()

# Department mapping
departments = {
    "ACCOUNT": "Account Support",
    "CANCEL": "Order Management",
    "CONTACT": "Customer Service",
    "DELIVERY": "Logistics",
    "FEEDBACK": "Customer Relations",
    "INVOICE": "Finance",
    "ORDER": "Order Management",
    "PAYMENT": "Finance",
    "REFUND": "Finance",
    "SHIPPING": "Logistics",
    "SUBSCRIPTION": "Subscription Support"
}

# Prediction function
def predict(text):

    inputs = tokenizer(text,padding="max_length",truncation=True,max_length=32,return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    prob = torch.softmax(outputs.logits, dim=1)
    #return value and label(index)
    confidence, predicted_label = prob.max(dim=1)

    category = model.config.id2label[predicted_label.item()]
    department = departments[category]

    return category, confidence.item() * 100, department


# User input
if __name__ == "__main__":

    text = input("Enter the complaint: ")

    category, confidence, department = predict(text)

    print("Category:", category)
    print(f"Confidence: {confidence:.2f}%")
    print("Department:", department)