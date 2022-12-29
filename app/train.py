import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import NeuralNet
from nltk_utils import tokenize, stemming, bag_of_words

with open("app/intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", ".", ",", "!", "/", "'s"]
all_words = [stemming(w) for w in all_words if w not in ignore_words]

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)   # No need of OnehotEncoding, CrossEntropy Loss is used here.

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
# Number of samples in the training data.
batch_size = 8

dataset = ChatDataset()
# if num_workers=2 shows error, then use 0 in windows.
# We have at most 2 workers simultaneously putting data into RAM (Multi-Processing/Multi-Threading)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


input_size = len(X_train[0])  # Or len(all_words)
hidden_size = 8
ouput_size = len(tags)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size=input_size,
                  hidden_size=hidden_size,
                  num_classes=ouput_size).to(device)

# loss and optimizer
learning_rate = 0.001
# Epochs - Number of complete passes through the training dataset.
num_epochs = 1000
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device=device, dtype=torch.int64)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}")

print(f"final loss, loss = {loss.item():.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": ouput_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

filename = "data.pth"
torch.save(data, filename)
print(f"Training complete, file saved as {filename}")