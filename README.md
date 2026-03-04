# Hinglish Delivery Assistant (Offline NLU)

An ultra lightweight Intent Parser (<1M params, <2MB) designed for low-end Android devices (2-4GB RAM) operating in offline environments.

## Approach: Hybrid NLU
Most Small Language Models (SLMs) struggle with Hinglish code-switching. I used a **Hybrid Architecture**:
1. **Neural Intent Classifier:** A Bidirectional GRU (Gated Recurrent Unit) processes the sequence to identify the user's goal.
2. **Heuristic Slot Extractor:** A rule-based engine handles entity extraction (Time, Order Reference) for 100% precision on business critical parameters.



## Technical Specifications
- **Architecture:** Bi-GRU + Global Max Pooling
- **Parameters:** ~650k
- **Model Size:** 650KB (.pth) / 630KB (ONNX)
- **Latency:** ~1ms on standard CPU
- **Vocabulary:** Custom Word level Tokenizer (No external dependencies used)

## Setup & Usage
1. Clone the project and make a virtual environment
2. Run `pip install requirements.txt`
3. Run `python train.py` to generate weights.
4. Run `python inference.py` to test commands.

## Qualitative Examples
| Input | Intent | Slots |
| :--- | :--- | :--- |
| "Bhai next order ka address batao" | `get_address` | `{"order": "next"}` |
| "Traffic ki wajah se 10 min late honga" | `report_delay` | `{"reason": "traffic", "delay_time": "10 mins"}` |
| "Sir pichla order cancel kar do" | `order_issue` | `{"order": "previous", "action": "cancellation"}` |
| "Customer phone nahi utha raha" | `customer_unavailable` | `{}` |
| "Map stuck ho gaya hai location do" | `navigation_help` | `{"item": "address"}` |

## Why Bidirectional GRU over Transformer model or LSTM model?
I chose GRU over Transformer because at <15M parameters, Transformers lack the global context needed for short sequences, while a Bi-GRU provides superior accuracy and 3x faster inference on mobile CPUs. I utilized a Bidirectional GRU instead of an LSTM to optimize for the hardware constraints of 2-4GB RAM devices. The GRU offers a more compact parameter set and lower computational overhead, which is critical for real-time offline inference without sacrificing NLU performance on short Hinglish command sequences.
