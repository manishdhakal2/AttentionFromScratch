# Attention Mechanisms from Scratch

This project contains implementations of different attention mechanisms used in modern deep learning architectures, built from scratch for learning and experimentation.

## Project Structure

### 1. `seq2seq_attention_numpy.py`
Implements a basic **Seq2Seq additive attention mechanism** using **NumPy**.

Features:
- Encoder-decoder attention
- Alignment score calculation
- Context vector generation
- Manual tensor operations for learning purposes

Purpose:
- Understand how classical attention works before transformers.

---

### 2. `attention.py`
Contains a **Self-Attention** class implemented with **PyTorch**.

Features:
- Query, Key, Value projections
- Scaled dot-product attention
- Softmax normalization
- Context vector generation

Purpose:
- Learn the core building block behind transformer models.

---

### 3. `multi_head_attention.py`
Implements **Multi-Head Attention** using PyTorch.

Features:
- Multiple attention heads
- Head splitting
- Parallel attention computation
- Head concatenation
- Final output projection

Purpose:
- Understand how transformers capture different relationships simultaneously.

---

## Concepts Covered

This project demonstrates:

- Seq2Seq attention
- Self-attention
- Scaled dot-product attention
- Multi-head attention
- Tensor reshaping
- Head splitting and recombination
- Attention score normalization

---

## Mathematical Formula

Scaled dot-product attention:

\[
Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Multi-head attention:

\[
MultiHead(Q,K,V)=Concat(head_1,...,head_h)W^O
\]

---

## Requirements

Install dependencies:

```bash
pip install numpy torch