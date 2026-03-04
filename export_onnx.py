import torch
from model import HinglishParser

def convert_to_onnx(model_path, vocab_size, num_intents, output_onnx="model.onnx"):
    # Initialize and load model
    model = HinglishParser(vocab_size=vocab_size, num_intents=num_intents, num_slots=5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create dummy input (Batch Size 1, Max Length 12)
    dummy_input = torch.randint(0, vocab_size, (1, 12))

    # Export
    torch.onnx.export(
        model, 
        dummy_input, 
        output_onnx,
        input_names=['input'], 
        output_names=['intent_logits', 'slot_logits'],
        dynamic_axes={'input': {0: 'batch_size'}}, # Allow different batch sizes
        opset_version=18
    )
    print(f"ONNX model saved to {output_onnx}")


convert_to_onnx("hinglish_parser.pth", vocab_size=137, num_intents=8)