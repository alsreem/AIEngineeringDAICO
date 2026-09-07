# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
## Lab 1 — Tokenizer Audit

| Tokenizer | Language | Samples | Fertility | Mean length | P95 length |
|---|---|---:|---:|---:|---:|
| mBERT | Arabic | 7200 | 2.153 | 18.98 | 27 |
| mBERT | English | 4800 | 1.510 | 16.48 | 25 |
| XLM-R | Arabic | 7200 | 1.672 | 15.19 | 21 |
| XLM-R | English | 4800 | 1.434 | 15.76 | 23 |
| CAMeLBERT | Arabic | 7200 | 1.405 | 13.08 | 20 |
| CAMeLBERT | English | 4800 | 2.705 | 27.94 | 38 |
| DistilBERT | Arabic | 7200 | 4.527 | 37.71 | 47 |
| DistilBERT | English | 4800 | 1.298 | 14.45 | 21 |

### Decision

XLM-R was selected as the most balanced bilingual tokenizer. CAMeLBERT has the best Arabic fertility, but its English fertility is substantially worse. DistilBERT performs well on English but poorly on Arabic. XLM-R provides the strongest overall Arabic/English balance with relatively low sequence lengths.
## Lab 2 — Transformer Anatomy

### Attention equivalence

| Check | Result |
|---|---:|
| Max absolute difference vs PyTorch | 0.0000002384 |
| atol requirement | 1e-6 |
| Numerical equivalence | PASS |

### Multi-Head Attention

| Input shape | Output shape |
|---|---|
| (1, 4, 8) | (1, 4, 8) |

### Parameter audit

| Checkpoint | Total params | Embeddings | Attention | FFN | Norms | Pooler |
|---|---:|---:|---:|---:|---:|---:|
| mBERT | 177,853,440 | 92,208,384 | 28,366,848 | 56,669,184 | 18,432 | 590,592 |
| CAMeLBERT | 109,081,344 | 23,436,288 | 28,366,848 | 56,669,184 | 18,432 | 590,592 |

### Causal mask

- Mask structure: lower triangular.
- Future positions blocked: `True`.
- Model family: Decoder-style causal attention.

### PAD attention leakage

| Configuration | PAD attention mass |
|---|---:|
| Without padding mask | 2.937145 |
| With padding mask | 0.000000 |

**Finding:** Without a correct padding mask, attention can leak onto PAD positions. Applying the padding mask reduced PAD attention mass to zero.


## Lab 3 — Models
| Model | Metric | Validation | Frozen test | Train time |
|---|---|---:|---:|---:|
| TF-IDF + LinearSVC | macro-F1 | | | |
| Topic classifier | macro-F1 | | | |
| NER | entity-F1 | | | |
| QA | span/null smoke | | | |

## Lab 4 — Arabic model bake-off
| Checkpoint | macro-F1 all | Gulf | MSA | AR fertility |
|---|---:|---:|---:|---:|
| multilingual incumbent | | | | |
| Arabic dialect-aware | | | | |
| optional third model | | | | |

## Lab 5 — Search
| Configuration | recall@10 | MRR@10 | p50 latency/query |
|---|---:|---:|---:|
| bi-encoder only | | | |
| + cross-encoder rerank | | | |
| cross-lingual slice | | | |

- no-answer empty-correct: ___ / 20
- cross-lingual gap: ___

## Lab 6 — Evaluation
| Model | Aggregate macro-F1 [CI] | Gulf [CI] | Invariance pass | MFT pass |
|---|---|---|---:|---:|
| topic classifier | | | | |
| dialect-aware | | | | |

- paired comparison verdict:
- error taxonomy top categories:
- top-3 prioritised fixes:

## Lab 7 — Optimisation ladder
| Rung | p50 | p99 | quality metric / paired Δ | Artefact size |
|---|---:|---:|---|---:|
| fp32 torch @512 padded | | | | |
| fp32 torch @128 dynamic | | | | |
| ONNX fp32 @128 | | | | |
| ONNX INT8 @128 | | | | |

- HTTP p99, 16 concurrent:
- classifier quantisation decision:
- NER quantisation decision:
