Aetheria — Model roadmap & prototype
=====================================

Goal (v1): a lightweight, embeddable "Aetheria" core — a small transformer
language model usable for game NPCs and research experiments on limited
hardware.

Design notes for v1 prototype:
- Modality: Text only (game/dialogue-centric). Later: adapters for other modalities.
- Size target: tiny (millions of parameters) to fit CPU / small GPU training and inference.
- Architecture: standard Transformer LM (small d_model, few layers, multi-head attention).
- Tokenizer: SentencePiece (BPE) trained on project data.

How to run prototype locally:
1. Prepare data: put cleaned conversations in `data/cleaned_conversations.txt` (one paragraph per conversation).
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run prototype trainer:

```bash
python scripts/prototype_model.py
```

Notes and next steps:
- With limited RAM and integrated GPU, training from scratch should start with small vocab (4k), short sequences, and few epochs.
- After validating training works, we'll iterate: smaller/better tokenizer, curriculum/data augmentation, quantized inference, and knowledge distillation to shrink the model.
- I'll add evaluation scripts, checkpointing, and a simple inference runner next.
