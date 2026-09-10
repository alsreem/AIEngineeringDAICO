# Decision Records

## tokenizer
- Chosen checkpoint(s): xlm-roberta-base
- Arabic fertility evidence: Lower subword fragmentation measured during the Lab 1 tokenizer audit compared to mBERT.
- English fertility evidence: Balanced and efficient tokenisation for bilingual feedback processing.
- p95 length evidence: Well within the sequence length constraints after analyzing the sample dataset.
- Operational trade-off / rationale: Chosen to handle code-switching between Arabic and English text robustly without exploding sequence lengths.

## arabic-model
- Incumbent: bert-base-multilingual-cased
- Candidate: CAMeL-Lab/bert-base-arabic-camelbert-mix
- All/Gulf/MSA evidence: CAMeLBERT-mix outperformed the baseline and general models on NER and QA tasks, specifically showing superior handling of the Gulf dialect slice during Lab 4 evaluation.
- CI-backed verdict: Validated through slice performance improvements in dialect-aware evaluation.
- Segmentation contract: Integrated clitic segmentation into the preprocessing and NER pipeline to boost LOCATION recall.

## search-min-score
- Threshold: 0.75 (Pending Lab 5 evaluation final tuning)
- No-answer evidence: Configured to support honest empty/no-result behaviour for out-of-domain queries.
- False-positive / false-negative trade-off: Tuned to minimise irrelevant retrieved historical citizen feedback cases.

## quantisation-split
- Topic artefact: Pending Lab 7 execution
- NER artefact: Pending Lab 7 execution
- Latency evidence: Pending CPU benchmark ladder
- Paired quality-tax evidence: Pending evaluation
- Rollback artefact retained: Yes (FP32 baseline kept)

## architecture
- Encoder/decoder rationale by task: Encoders (BERT/CAMeLBERT) used for classification, NER, and semantic search matching the Bayan service requirements.
- Multilingual vs Arabic-centric rationale: Shifted to an Arabic-centric model (CAMeLBERT) to better capture dialectal nuances in citizen feedback.
- Evidence used: Lab 1 tokenizer metrics, Lab 3 validation performance, and Lab 4 Arabic bake-off slice evaluations.
-
-
-
-
