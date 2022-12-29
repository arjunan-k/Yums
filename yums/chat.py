import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("intents.json", "r") as f:
    intents = json.load(f)

filename = "data.pth"
data = torch.load(filename)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)
# Loading learned parameters
model.load_state_dict(model_state)
# Setting to evaluation mode
model.eval()

bot_name = "Arjunan K"
def get_response(msg):
    sentence = tokenize(msg)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    # Conerting to torch tensor.
    x = torch.from_numpy(x).to(device)

    output = model(x)
    # Return max of the array vector from each row (dim=1)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    else:
        return "I do not understand..."