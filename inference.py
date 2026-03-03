import torch
import json
from model import HinglishParser  # Import the class from your model file
from tokenizer import HinglishTokenizer # Import your custom tokenizer

class DeliveryAssistant:
    def __init__(self, model_path, vocab_path):
        # Load Vocabulary and Metadata
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        self.tokenizer = HinglishTokenizer(max_length=vocab_data['max_length'])
        self.tokenizer.word2idx = vocab_data['word2idx']
        self.tokenizer.intent2idx = vocab_data['intent2idx']
        self.tokenizer.idx2intent = {v: k for k, v in vocab_data['intent2idx'].items()}
        
        # Initialize Model Architecture
        self.model = HinglishParser(
            vocab_size=len(self.tokenizer.word2idx),
            num_intents=len(self.tokenizer.intent2idx),
            num_slots=5
        )
        
        #Load Trained Weights
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval() #Set to evaluation mode

    def parse(self, text):
        with torch.no_grad():
            # Predict Intent using the GRU Model
            tokens = torch.tensor([self.tokenizer.encode(text)])
            intent_logits, _ = self.model(tokens)
            intent_idx = torch.argmax(intent_logits, dim=1).item()
            intent_name = self.tokenizer.idx2intent[intent_idx]
            
            # Extract Slots using Keyword Heuristics
            slots = {}
            text_lower = text.lower()
            
            # Order References
            if any(word in text_lower for word in ["next", "agla", "doosra"]):
                slots["order"] = "next"
            elif any(word in text_lower for word in ["pichla", "previous", "last"]):
                slots["order"] = "previous"
                
            # Delay Reasons & Time
            if "traffic" in text_lower:
                slots["reason"] = "traffic"
            if "puncture" in text_lower or "bike" in text_lower:
                slots["reason"] = "vehicle_issue"
                
            # Extract Time (Simple regex-style check)
            import re
            time_match = re.search(r'(\d+)\s*(min|minute|m)', text_lower)
            if time_match:
                slots["delay_time"] = f"{time_match.group(1)} mins"
                
            # Issues
            if any(word in text_lower for word in ["cancel", "radd"]):
                slots["action"] = "cancellation"
            if "missing" in text_lower:
                slots["issue"] = "item_missing"

            return {"text": text, "intent": intent_name, "slots": slots}

if __name__ == "__main__":
    assistant = DeliveryAssistant("hinglish_parser.pth", "vocab.json")
    
    # Test a few commands
    commands = [
        "Bhai next order ka address batao",
        "Customer call nahi utha raha",
        "Traffic ki wajah se 10 min late honga",
        "Sir pichla order cancel kar do",
        "10 min late honga",
        "Traffic ki wajah se late hu"
    ]
    
    for cmd in commands:
        result = assistant.parse(cmd)
        print(f"Input: {result['text']}")
        print(f"Output: {json.dumps(result, indent=2)}\n")