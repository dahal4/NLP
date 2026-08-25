# Chapter 3 — Skip-gram with Negative Sampling (Word2Vec)

Implement **Word2Vec SGNS** in pure Python. The only numeric library is **NumPy** — no PyTorch, TensorFlow, or Keras.

## Setup

```bash
pip install numpy matplotlib
```

## Files

| File | Description |
|------|-------------|
| `sgns_word2vec.py` | Full SGNS implementation + toy-corpus demo |
| `Assignment3.ipynb` | Step-by-step walkthrough of the same algorithm |

## Run

```bash
python Chapter3/sgns_word2vec.py
```

Or open `Assignment3.ipynb` and run all cells.

## What it does

1. **Tokenize** — Lowercase text, keep alphabetic words.
2. **Skip-gram pairs** — For each center word, take neighbors inside a window as positive context.
3. **Negative samples** — Draw `k` random words from the unigram^{0.75} noise distribution.
4. **SGD** — Each pair is logistic regression: push the true neighbor together, push the random words apart.
5. **Query** — Cosine nearest neighbors of the center embeddings `W_in`.
6. **Live plot** — After every epoch, redraw training loss (should fall) and a 2D PCA of the word vectors (related words should cluster).

The toy corpus is tiny on purpose (royalty vs pets), so training finishes in a few seconds and neighbors like `king → queen, prince, kingdom` are easy to inspect.
