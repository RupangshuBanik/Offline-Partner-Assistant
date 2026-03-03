import json
import re
from collections import Counter

class HinglishTokenizer:
    def __init__(self, max_length=15):
        self.max_length = max_length
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        self.intent2idx = {}
        self.idx2intent = {}

    def _tokenize(self, text):
        # Basic normalization: lowercase and remove special chars
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def build_vocab(self, data):
        """
        data: list of dicts {"text": "...", "intent": "..."}
        """
        all_words = []
        intents = set()
        
        for item in data:
            words = self._tokenize(item['text'])
            all_words.extend(words)
            intents.add(item['intent'])
        
        # Build word vocabulary
        word_counts = Counter(all_words)
        # We include all words since our dataset is small and specific
        for word, _ in word_counts.most_common():
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        
        # Build intent/label mapping
        for i, intent in enumerate(sorted(list(intents))):
            self.intent2idx[intent] = i
            self.idx2intent[i] = intent
            
        print(f"Vocab size: {self.vocab_size}")
        print(f"Number of intents: {len(self.intent2idx)}")

    def encode(self, text):
        """Converts string to a list of IDs with padding"""
        tokens = self._tokenize(text)
        token_ids = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokens]
        
        # Truncate or Pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids += [self.word2idx["<PAD>"]] * (self.max_length - len(token_ids))
            
        return token_ids

    def save_vocab(self, filepath):
        with open(filepath, 'w') as f:
            json.dump({
                "word2idx": self.word2idx,
                "intent2idx": self.intent2idx,
                "max_length": self.max_length
            }, f)



if __name__ == "__main__":
    with open("delivery_data.json", "r") as f:
        raw_data = json.load(f)
    
    tokenizer = HinglishTokenizer(max_length=15) # Average delivery partner command is shorter than 15 words
    tokenizer.build_vocab(raw_data)
    
    #Testing
    test_phrase = "Bhaiya location bhejo jaldi"
    encoded = tokenizer.encode(test_phrase)
    print(f"\nTest Phrase: {test_phrase}")
    print(f"Encoded IDs: {encoded}")
    
    #Save for training phase
    tokenizer.save_vocab("vocab.json")