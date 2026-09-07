# Lab Notes

## Lab 1 — Defect Safari
Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1
- Class: Leading/trailing and repeated whitespace
- Example: `  ألعاب الأطفال في حديقة حي العليا تحتاج صيانة   `
- Why it matters: Extra whitespace creates inconsistent text representations and can affect tokenisation and evaluation.
- Decision: Clean — collapse repeated whitespace and strip leading/trailing whitespace.

### Defect 2
- Class: Arabic Tatweel
- Example: `الخدمــــة`
- Why it matters: Tatweel is a presentation character that can increase token fragmentation without adding task meaning.
- Decision: Clean — remove Tatweel.

### Defect 3
- Class: Character repetition / elongation
- Example: `لووووسمحت`
- Why it matters: Excessive repeated characters increase vocabulary fragmentation and sequence length while usually preserving the same meaning.
- Decision: Clean — limit repeated characters to two.

### Defect 4
- Class: Personally identifiable information (PII)
- Example: `0551234567 1023456789`
- Why it matters: Phone numbers and national IDs are sensitive personal information and should not be exposed to downstream models.
- Decision: Clean/mask — replace phone numbers with `<PHONE>` and national IDs with `<NATIONAL_ID>`.

### Defect 5
- Class: HTML markup
- Example: `<br>`
- Why it matters: Markup is formatting rather than citizen language and can introduce unnecessary tokens.
- Decision: Clean — remove HTML markup before model processing.

### Defect 6
- Class: Duplicate/repeated words
- Example: `My My Licence licence request`
- Why it matters: Accidental word duplication can increase token count and may distort model predictions if treated as meaningful content.
- Decision: Task-dependent — preserve unless there is evidence that the repetition is a data-entry artefact that should be corrected.


## Lab 2 — Parameter audit
| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | | | |
| CAMeLBERT | | | |

## Lab 4 — Dialect audit
- Distribution:
- One-sentence implication for MSA-only evaluation:
