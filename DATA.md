# Data Strategy Document

## Sourcing & Generation
Since no public dataset exists for "Hinglish Delivery Partner Commands," I developed a **Synthetic Data Pipeline** (refer to `data_gen.py`).
- **Templates:** 8 core intents with 100 variations each.
- **Diversity:** Included "Bhai/Sir" prefixes, regional dialects ("Pata", "Radd"), and common typos ("Deliverd", "Restrunt").
- **Noise Injection:** Simulated low connectivity typing errors to improve model robustness.

## Preprocessing
1. **Normalization:** Lowercasing and regex-based punctuation removal.
2. **Tokenization:** Custom Word-to-Idx mapping to keep the embedding table minimal.
3. **Padding:** Fixed `max_length=15` to ensure constant memory footprint on mobile.