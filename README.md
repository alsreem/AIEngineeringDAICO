
# Bayan | بيان
## SDA-AIE-211 — Natural Language Processing with Transformers

> **From raw bilingual text to a working NLP service — one lab at a time.**
>
> في هذا المشروع ما راح نبني 7 تمارين منفصلة. راح نطوّر **Bayan** خطوة بخطوة: من raw Arabic/English text، إلى preprocessing وTransformers وfine-tuning وsemantic search وevaluation، ثم نختم بخدمة FastAPI محسّنة وقابلة للقياس.

---

# 🚀 What are we building?

**Bayan (بيان)** is a bilingual citizen-feedback intelligence service for Arabic and English text.

By the end of the course, this repository should be able to:

- 🧹 preprocess Arabic + English text consistently
- 🔐 mask PII before model use
- 🧠 classify feedback topics/sentiment
- 🏷️ extract entities with NER
- ❓ handle extractive QA with honest no-answer behaviour
- 🔎 retrieve similar historical cases using semantic search
- 📊 evaluate models with slices, confidence intervals, and behavioural tests
- ⚡ optimise inference with ONNX + INT8
- 🌐 serve the final pipeline through FastAPI
## 🛠️ What We Have Built So Far (Labs 1–4)

- **Lab 1 (Bilingual Preprocessing & Tokenizer Audit):** Built Bayan's shared bilingual preprocessing module, ensured 100% PII masking recall, handled sentence segmentation for noisy citizen feedback, compared multiple tokenizers (including mBERT and XLM-R), and made an evidence-based tokenizer decision[cite: 1].
- **Lab 2 (Anatomy of a Transformer):** Implemented scaled dot-product attention and Multi-Head Attention from scratch, verified numerical equivalence against PyTorch, audited parameter counts across layers, built causal masks, and diagnosed attention-mask leakage[cite: 1].
- **Lab 3 (Topic Classification, NER & Extractive QA):** Established a TF-IDF baseline, built leakage-safe grouped splits with zero citizen overlap, fine-tuned a topic classifier beating the baseline, aligned subword BIO labels for NER, and built extractive QA post-processing with honest no-answer handling[cite: 1].
- **Lab 4 (Arabic Pipeline & Dialect-Aware Fine-Tuning):** Implemented Arabic normalisation profiles, audited dialect distributions across slices, integrated clitic segmentation into the NER pipeline to boost location recall, and conducted an Arabic model bake-off where the dialect-aware model (CAMeLBERT) outperformed baselines on the Gulf slice[cite: 1].

https://github.com/SDAIAAcademy
 
