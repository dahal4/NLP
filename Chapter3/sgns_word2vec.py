"""Skip-gram with Negative Sampling (SGNS) Word2Vec.

Pure Python + NumPy. No deep learning frameworks.

SGNS trains two embedding tables so that a center word is close to the
words that actually appear around it, and far from randomly sampled
(negative) words.
"""

import re
from collections import Counter

import numpy as np


def sigmoid(x):
    """Return 1 / (1 + exp(-x)), clipped so exp does not overflow."""
    x = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-x))


def tokenize(text):
    """Lowercase `text` and return alphabetic word tokens."""
    return re.findall(r"[a-z]+", text.lower())


def build_vocab(tokens, min_count=1):
    """Build word-id maps and a count vector from a token list.

    Parameters
    ----------
    tokens : list of str
        Tokenized corpus.
    min_count : int
        Drop words that occur fewer than this many times.

    Returns
    -------
    word2id : dict[str, int]
    id2word : dict[int, str]
    counts : ndarray of shape (vocab_size,)
        Raw unigram counts, aligned with ids.
    """
    counter = Counter(tokens)
    words = [w for w, c in counter.most_common() if c >= min_count]
    word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for w, i in word2id.items()}
    counts = np.array([counter[w] for w in words], dtype=np.float64)
    return word2id, id2word, counts


def skipgram_pairs(token_ids, window):
    """Build (center, context) id pairs with a symmetric context window.

    For the sequence [a, b, c] and window=1 the pairs are
    (a, b), (b, a), (b, c), (c, b).
    """
    pairs = []
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        left = max(0, i - window)
        right = min(n, i + window + 1)
        for j in range(left, right):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs


def noise_distribution(counts, power=0.75):
    """Return P(w) proportional to count(w)^power (Mikolov et al.).

    The 0.75 exponent makes rare words a bit more likely to be drawn
    as negatives than a raw unigram would.
    """
    probs = np.power(counts, power)
    probs /= probs.sum()
    return probs


def sample_negatives(context_id, k, noise_probs, rng):
    """Draw `k` unique negative word ids, excluding the true context."""
    vocab_size = len(noise_probs)
    k = min(k, vocab_size - 1)
    negs = []
    seen = {context_id}
    while len(negs) < k:
        word_id = int(rng.choice(vocab_size, p=noise_probs))
        if word_id not in seen:
            seen.add(word_id)
            negs.append(word_id)
    return negs


class SkipGramNS:
    """Skip-gram Word2Vec trained with negative sampling.

    Each training step is logistic regression on 1 + k examples:

    * label 1 — the true (center, context) pair
    * label 0 — k randomly drawn (center, negative) pairs

    The vectors we usually keep afterwards are the center rows `W_in`.
    """

    def __init__(self, vocab_size, dim=50, lr=0.05, seed=0):
        """Initialize small random center and context embeddings.

        Parameters
        ----------
        vocab_size : int
            Number of distinct words.
        dim : int
            Embedding dimension.
        lr : float
            SGD learning rate.
        seed : int
            RNG seed for reproducible init.
        """
        rng = np.random.RandomState(seed)
        scale = 0.5 / dim
        self.W_in = rng.uniform(-scale, scale, size=(vocab_size, dim))
        self.W_out = rng.uniform(-scale, scale, size=(vocab_size, dim))
        self.lr = lr
        self.vocab_size = vocab_size

    def train_example(self, center, targets, labels):
        """One SGD step for a center word vs. one positive and k negatives.

        Parameters
        ----------
        center : int
            Center-word id.
        targets : ndarray of shape (1 + k,)
            True context id followed by negative ids.
        labels : ndarray of shape (1 + k,)
            1.0 for the true context, 0.0 for negatives.

        Returns
        -------
        float
            Binary cross-entropy loss for this example.
        """
        h = self.W_in[center].copy()
        v = self.W_out[targets]
        scores = v @ h
        preds = sigmoid(scores)
        error = preds - labels

        self.W_in[center] -= self.lr * (error @ v)
        self.W_out[targets] -= self.lr * np.outer(error, h)

        eps = 1e-12
        return float(
            -np.sum(
                labels * np.log(preds + eps)
                + (1.0 - labels) * np.log(1.0 - preds + eps)
            )
        )

    def most_similar(self, word, word2id, id2word, topn=5):
        """Return the `topn` nearest words by cosine similarity of `W_in`."""
        if word not in word2id:
            return []
        idx = word2id[word]
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        unit = self.W_in / (norms + 1e-12)
        sims = unit @ unit[idx]
        sims[idx] = -np.inf
        ranked = np.argsort(-sims)[:topn]
        return [(id2word[i], float(sims[i])) for i in ranked]


def train(model, pairs, noise_probs, k, epochs, rng, verbose=True, on_epoch=None):
    """Shuffle pairs each epoch and run SGD with `k` negatives per pair.

    Parameters
    ----------
    on_epoch : callable or None
        If given, called as `on_epoch(epoch, mean_loss, history, model)`
        after every epoch (use this to redraw a learning plot).

    Returns
    -------
    list of float
        Mean loss per epoch.
    """
    history = []
    pairs = list(pairs)
    for epoch in range(epochs):
        rng.shuffle(pairs)
        total = 0.0
        for center, context in pairs:
            negs = sample_negatives(context, k, noise_probs, rng)
            labels = np.array([1.0] + [0.0] * len(negs))
            targets = np.array([context] + negs)
            total += model.train_example(center, targets, labels)
        mean_loss = total / max(len(pairs), 1)
        history.append(mean_loss)
        if on_epoch is not None:
            on_epoch(epoch, mean_loss, history, model)
        elif verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print("epoch {:3d}  loss {:.4f}".format(epoch + 1, mean_loss))
    return history


def self_check():
    """Fail fast if pair building, sigmoid, or SGD is broken."""
    pairs = skipgram_pairs([0, 1, 2], window=1)
    assert pairs == [(0, 1), (1, 0), (1, 2), (2, 1)], pairs

    sig = sigmoid(np.array([0.0, 100.0, -100.0]))
    assert abs(sig[0] - 0.5) < 1e-9
    assert sig[1] > 0.99 and sig[2] < 0.01

    tokens = tokenize("alpha beta gamma alpha beta gamma " * 30)
    word2id, _, counts = build_vocab(tokens)
    token_ids = [word2id[t] for t in tokens]
    pairs = skipgram_pairs(token_ids, window=1)
    model = SkipGramNS(len(word2id), dim=8, lr=0.1, seed=1)
    rng = np.random.RandomState(1)
    history = train(
        model, pairs, noise_distribution(counts), k=2, epochs=8, rng=rng, verbose=False
    )
    assert history[-1] < history[0], (history[0], history[-1])
    print("self-check passed")


TOY_CORPUS = """
the king rules the kingdom
the queen rules the kingdom
the king and the queen live in the palace
a man is a king
a woman is a queen
the man lives in the house
the woman lives in the house
cats drink milk
dogs chew bones
a cat is a pet
a dog is a pet
cats and dogs live in the house
the prince is a young king
the princess is a young queen
"""


def project_2d(W):
    """Project embedding rows to 2D with PCA (NumPy only)."""
    centered = W - W.mean(axis=0, keepdims=True)
    if W.shape[1] < 2:
        pad = np.zeros((W.shape[0], 2))
        pad[:, : W.shape[1]] = centered
        return pad
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return centered @ eigvecs[:, np.argsort(eigvals)[-2:]]


def make_live_plot(id2word):
    """Return an on_epoch callback that redraws loss + word vectors."""
    import matplotlib.pyplot as plt

    plt.ion()
    fig, (ax_loss, ax_emb) = plt.subplots(1, 2, figsize=(12, 5))

    def show_learning(epoch, mean_loss, history, model):
        """Redraw both axes after one training epoch."""
        ax_loss.clear()
        ax_emb.clear()
        epochs_x = list(range(1, len(history) + 1))
        ax_loss.plot(epochs_x, history, color="C0")
        ax_loss.scatter([epoch + 1], [mean_loss], color="C1", zorder=3)
        ax_loss.set_title("Training loss (falling = learning)")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Mean binary cross-entropy")
        ax_loss.grid(True, alpha=0.3)

        xy = project_2d(model.W_in)
        ax_emb.scatter(xy[:, 0], xy[:, 1], s=30, color="C0")
        for i, (x, y) in enumerate(xy):
            ax_emb.annotate(
                id2word[i],
                (x, y),
                fontsize=9,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax_emb.set_title("Word vectors after epoch {}".format(epoch + 1))
        ax_emb.set_xlabel("PC1")
        ax_emb.set_ylabel("PC2")
        ax_emb.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()
        plt.pause(0.01)
        print("epoch {:3d}  loss {:.4f}".format(epoch + 1, mean_loss))

    return fig, show_learning


def main():
    """Train SGNS on a tiny corpus and print nearest neighbors."""
    self_check()

    tokens = tokenize(TOY_CORPUS)
    word2id, id2word, counts = build_vocab(tokens)
    token_ids = [word2id[t] for t in tokens]
    pairs = skipgram_pairs(token_ids, window=2)

    print("vocab size:", len(word2id))
    print("training pairs:", len(pairs))

    rng = np.random.RandomState(0)
    model = SkipGramNS(vocab_size=len(word2id), dim=30, lr=0.05, seed=0)
    fig, show_learning = make_live_plot(id2word)
    train(
        model,
        pairs,
        noise_distribution(counts),
        k=5,
        epochs=80,
        rng=rng,
        on_epoch=show_learning,
    )

    for query in ("king", "queen", "cat", "dog", "house"):
        neighbors = model.most_similar(query, word2id, id2word, topn=5)
        pretty = ", ".join("{0} ({1:.2f})".format(w, s) for w, s in neighbors)
        print("{0:8s} -> {1}".format(query, pretty))

    import matplotlib.pyplot as plt

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
