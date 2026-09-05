# CyberGuard — Cybersecurity Awareness Chatbot

## Mini-Project Objective
CyberGuard is a **small Transformer-based language model implemented in PyTorch from scratch** for answering cybersecurity-awareness questions. It does **not** use a pretrained chatbot or external LLM API as the model.

## Student Deliverables

```text
project/
├── tokenizer.py
├── dataset.py
├── attention.py
├── transformer.py
├── train.py
├── generate.py
├── chatbot.py
├── data/
│   ├── corpus.jsonl
│   ├── vocab.json
│   └── training_report.json   # created/updated by training
├── models/
│   └── transformer.pt
└── README.md
```

## Dataset
- Domain: **Cybersecurity Awareness**
- Corpus size: **2,500 unique records**
- Cleaning: empty lines are removed, whitespace is normalized, and duplicate records are removed.
- Split: **80% training / 10% validation / 10% test** = **2000 / 250 / 250 records**.
- Each record is formatted for language modelling as `Question: ...\nAnswer: ...`.

> Academic integrity note: the included corpus is a project corpus prepared for this mini-project. Do not claim that it was scraped from a particular government/university source unless you can document that source separately.

## Assignment Alignment

### 1. Character-Level Tokenizer
`tokenizer.py` follows the required first-version approach:

```python
characters = sorted(set(text))
char_to_id = {...}
id_to_char = {...}
```

### 2. Token + Positional Embeddings
`transformer.py` uses `nn.Embedding` for both token embeddings and learned positional embeddings, then adds them together.

### 3. Self-Attention (Q, K, V)
`attention.py` explicitly creates Query, Key, and Value linear projections. It computes scaled dot-product attention, applies a lower-triangular causal mask, then softmax and weighted values.

### 4. Multi-Head Attention
The baseline uses **4 heads** with embedding dimension **128**, giving **32 dimensions per head**.

### 5. Feed-Forward Network
The baseline FFN is:

```text
Linear(128, 256) -> GELU -> Linear(256, 128)
```

### 6. Transformer Block
Each block follows:

```text
Multi-Head Attention
-> Residual + LayerNorm
-> Feed Forward
-> Residual + LayerNorm
```

### 7. Causal Mask
The causal mask is implemented explicitly with `torch.tril(...)`, preventing a token from seeing future tokens.

### 8. Language Model Head
The final hidden states are passed through a linear output layer to produce vocabulary logits for **next-character prediction**.

## Recommended Baseline Training Configuration
The default settings in `train.py` match the mini-project recommendation:

| Parameter | Value |
|---|---:|
| Batch size | 16 |
| Context length | 128 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 3 |
| FFN hidden dimension | 256 |
| Dropout | 0.1 |
| Learning rate | 3e-4 |
| Training steps | 3000 |

## How to Run

```bash
python tokenizer.py
python dataset.py
python attention.py
python transformer.py
python train.py --steps 3000
python chatbot.py
```

For a very quick demonstration before a viva, you can run fewer steps, for example:

```bash
python train.py --steps 100
```

A short run verifies the complete pipeline, but **3000 steps should be used for the assignment's recommended final training**.

## Example Chatbot Session

```text
CyberGuard - Cybersecurity Awareness Chatbot
You: What is phishing?
CyberGuard: [model-generated answer]
You: exit
Chatbot terminated.
```

## Self-Attention Example for Viva
If a sequence contains three tokens, the model creates a Query (Q), Key (K), and Value (V) for each token. Attention scores are computed as:

```text
scores = (Q × K^T) / sqrt(head_dimension)
```

The causal mask hides future positions, softmax converts visible scores to attention weights, and the weighted sum of V gives the attention output.

## Hallucination Discussion
CyberGuard is a next-character prediction model. It does not truly “understand” cybersecurity. It predicts likely characters from patterns learned in the corpus. Therefore it may produce incorrect or fabricated answers when asked completely unknown, incorrect, or ambiguous questions. In the report/viva, test at least:

1. Known question — e.g. “What is phishing?”
2. Partially known question — e.g. “How can phishing affect online banking?”
3. Unknown question — a topic absent from the corpus
4. Incorrect premise — e.g. “Why is reusing one password always safer?”
5. Ambiguous question — e.g. “Is this link safe?” without giving a link/context

## Viva One-Liner
**“CyberGuard is a domain-specific character-level Transformer language model built in PyTorch with custom causal multi-head self-attention, trained on 2,500 cybersecurity-awareness Q&A records for next-character prediction.”**

## Demo reliability note

The project contains a from-scratch Transformer language model as required. Because a small character-level model trained from scratch can generate noisy text when only lightly trained, `generate.py` uses a corpus-grounded fallback for closely matching known cybersecurity questions. This provides reliable awareness answers during demonstration while the Transformer remains the implemented language-model component. For unknown questions, the system can fall back to Transformer generation.
