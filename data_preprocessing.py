from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader

dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset",split="train")

df = dataset.to_pandas()
df = df[["instruction", "category"]] 
df = df.dropna()
df = df.drop_duplicates()

# Label encoding
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["category"])

# Split dataset
train_data, temp_data = train_test_split(df,test_size=0.2,random_state=42,stratify=df["label"])
val_data, test_data = train_test_split(temp_data,test_size=0.5,random_state=42,stratify=temp_data["label"])

train_data, _ = train_test_split(train_data, train_size=9000, random_state=42, stratify=train_data["label"])
val_data, _ = train_test_split(val_data, train_size=1000, random_state=42, stratify=val_data["label"])
test_data, _ = train_test_split(test_data, train_size=1000, random_state=42, stratify=test_data["label"])
# print("Training samples:", len(train_data))
# print("Validation samples:", len(val_data))
# print("Testing samples:", len(test_data))

# print(df.head())
# print("Dataset shape:", df.shape)
# print("Label mapping:", dict(zip(encoder.classes_, range(len(encoder.classes_)))))

model_name = "distilbert/distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_inputs = tokenizer(    train_data["instruction"].tolist(),padding="max_length",truncation=True,max_length=32)
val_inputs = tokenizer(val_data["instruction"].tolist(),padding="max_length",truncation=True,max_length=32)
test_inputs = tokenizer( test_data["instruction"].tolist(), padding="max_length",truncation=True, max_length=32)

# print("Input IDs:", train_inputs["input_ids"][0])
# print("Attention mask:", train_inputs["attention_mask"][0])
# print("Label:", train_data["label"].iloc[0])
# print("Sequence length:", len(train_inputs["input_ids"][0]))

train_dataset = TensorDataset(torch.tensor(train_inputs["input_ids"]),torch.tensor(train_inputs["attention_mask"]),torch.tensor(train_data["label"].tolist()))
val_dataset = TensorDataset(torch.tensor(val_inputs["input_ids"]),torch.tensor(val_inputs["attention_mask"]),torch.tensor(val_data["label"].tolist()))
test_dataset = TensorDataset(torch.tensor(test_inputs["input_ids"]),torch.tensor(test_inputs["attention_mask"]),torch.tensor(test_data["label"].tolist()))


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

