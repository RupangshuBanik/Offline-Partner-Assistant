import torch
import torch.nn as nn

class HinglishParser(nn.Module):
    def __init__(self, vocab_size, num_intents, num_slots, embedding_dim=64, hidden_dim=128):
        super(HinglishParser, self).__init__()
        
        #Embedding Layer: Turns word IDs into vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        #GRU Layer: Processes the sequence (Bidirectional for better context)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        #Intent Head: Takes the final hidden state to classify the whole sentence
        #(hidden_dim * 2 because it's bidirectional)
        self.intent_classifier = nn.Linear(hidden_dim * 2, num_intents)
        
        #Slot Head: Predicts a tag for EACH word in the sentence
        self.slot_classifier = nn.Linear(hidden_dim * 2, num_slots)

    def forward(self, x):
        #x shape: (batch_size, max_length)
        embedded = self.embedding(x) # (batch_size, max_length, embedding_dim)
        
        #gru_out: (batch_size, max_length, hidden_dim * 2)
        #hidden: (num_layers * 2, batch_size, hidden_dim)
        gru_out, hidden = self.gru(embedded)
        
        #For intent, we take the last hidden state (concatenating forward and backward)
        #Or more simply: pooling across the sequence
        last_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        intent_logits = self.intent_classifier(last_hidden)
        
        #For slots, we use the output at every time step
        slot_logits = self.slot_classifier(gru_out)
        
        return intent_logits, slot_logits