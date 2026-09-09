# Model Benchmarks & Parameter Audit

## Parameter Distribution Comparison

| Model Architecture | Embedding Params | Attention Params | FFN Params | Total Parameters |
| :--- | :--- | :--- | :--- | :--- |
| `bert-base-multilingual-cased` | 92,208,384 | 28,366,848 | 56,669,184 | 177,853,440 |
| `CAMeL-Lab/bert-base-arabic-camelbert-mix` | 23,436,288 | 28,366,848 | 56,669,184 | 109,081,344 |

## Summary Notes
- The multilingual model requires significantly larger embedding layers (~92.2M) to handle a diverse vocabulary.
- Specialized Arabic models like `camelbert-mix` optimize the vocabulary size down to ~23.4M embeddings, substantially reducing total memory footprint while maintaining high domain capacity.
