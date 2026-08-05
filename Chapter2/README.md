# Chapter 2 — Assignment 2
Hands-on exercises for word frequency, vocabulary, co-occurrence, smoothing, next-word prediction, and TF-IDF — using plain Python (`re`, `collections`, `math`).

## What it does

1. **Word frequency counter** — Lowercase text, tokenize with `\b\w+\b`, count word occurrences in a dictionary.
2. **Vocabulary builder** — Split text into sentences, collect unique tokens into a set, report vocabulary size.
3. **Co-occurrence matrix** — On the first sentence, build a window-size-1 neighbor matrix over ordered unique vocabulary.
4. **Data sparsity** — From the co-occurrence matrix, list unique word pairs that never appear as neighbors.
5. **Laplace smoothing** — Unigram probabilities: `(count + 1) / (total_count + vocab_size)`.
6. **Keyboard prediction** — Bigram counts over a small corpus; suggest top-3 next words for `"I like"`.
7. **TF-IDF** — Compute TF, DF, IDF (`log(N / df)`), and TF-IDF for a three-document corpus.

Note : Sample input in early exercises is excerpted from a [News Decoder](https://news-decoder.com/) article on lethal autonomous weapons systems.

