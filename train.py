import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizer import HinglishTokenizer
from model import HinglishParser
import json

#Simple Dataset Class
class DeliveryDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x = torch.tensor(self.tokenizer.encode(item['text']))
        y_intent = torch.tensor(self.tokenizer.intent2idx[item['intent']])
        return x, y_intent

#Training Loop
def train_model(data_path, vocab_path):
    #Setup (Assuming your tokenizer is already built)
    #Import your class
    
    with open("delivery_data.json", 'r') as f:
        raw_data = json.load(f)
        
    tokenizer = HinglishTokenizer(max_length=15)
    tokenizer.build_vocab(raw_data)
    
    dataset = DeliveryDataset("delivery_data.json", tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = HinglishParser(
        vocab_size=tokenizer.vocab_size, 
        num_intents=len(tokenizer.intent2idx),
        num_slots=5
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #Loop
    print("Starting training...")
    model.train()
    for epoch in range(30):
        total_loss = 0
        for texts, intents in loader:
            optimizer.zero_grad()
            
            intent_pred, _ = model(texts)
            loss = criterion(intent_pred, intents)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader)}")

    #Save weights
    torch.save(model.state_dict(), "hinglish_parser.pth")
    print("Model saved as hinglish_parser.pth")

if __name__ == "__main__":
    train_model("delivery_data.json", "vocab.json")