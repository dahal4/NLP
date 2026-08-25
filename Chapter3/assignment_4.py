import os
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to load IMDB dataset (very simple: expects 'pos' and 'neg' folders of text files)
def load_imdb_data(imdb_dir, max_per_class=2500):
    texts = []
    labels = []
    for label, subdir in enumerate(['neg', 'pos']):
        dir_path = os.path.join(imdb_dir, 'train', subdir)
        files = [fname for fname in os.listdir(dir_path) if fname.endswith('.txt')][:max_per_class]
        for fname in files:
            with open(os.path.join(dir_path, fname), encoding='utf-8') as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels

# Text tokenization
def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

# Build vocabulary from training data
def build_vocab(texts, min_count=2):
    tokens = []
    for text in texts:
        tokens.extend(tokenize(text))
    counter = Counter(tokens)
    words = [w for w, c in counter.items() if c >= min_count]
    word2id = {w: i for i, w in enumerate(words)}
    return word2id

# Load pre-trained GloVe embeddings (download e.g., glove.6B.100d.txt beforehand)
def load_glove(glove_path, word2id, dim=100):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2id), dim))
    found = 0
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            if word in word2id:
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word2id[word]] = vector
                found += 1
    print(f"Loaded {found} GloVe vectors for vocab words.")
    return embeddings

# Convert a text to a vector (averaged word embeddings)
def text_to_vector(text, word2id, embeddings):
    tokens = tokenize(text)
    vecs = []
    for token in tokens:
        idx = word2id.get(token)
        if idx is not None:
            vecs.append(embeddings[idx])
    if len(vecs) == 0:
        return np.zeros(embeddings.shape[1])
    return np.mean(vecs, axis=0)

# Bag-of-words features (baseline)
def text_to_bow(text, word2id):
    tokens = tokenize(text)
    vec = np.zeros(len(word2id))
    for token in tokens:
        if token in word2id:
            vec[word2id[token]] += 1
    return vec

# Main workflow
def main():
    # Paths - adapt these to your environment
    imdb_dir = './aclImdb' # Download and extract IMDB dataset: https://ai.stanford.edu/~amaas/data/sentiment/
    glove_path = './glove.6B/glove.6B.100d.txt' # Download from https://nlp.stanford.edu/projects/glove/
    embedding_dim = 100

    # 1. Load and split data
    texts, labels = load_imdb_data(imdb_dir, max_per_class=2000)
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 2. Build vocabulary from train
    word2id = build_vocab(X_train, min_count=2)

    print("Vocab size:", len(word2id))

    # 3. Bag-of-words baseline features
    print("\nBag-of-words baseline:")
    X_train_bow = np.array([text_to_bow(t, word2id) for t in X_train])
    X_test_bow = np.array([text_to_bow(t, word2id) for t in X_test])
    clf_bow = LogisticRegression(max_iter=200)
    clf_bow.fit(X_train_bow, y_train)
    preds_bow = clf_bow.predict(X_test_bow)
    print("Test Accuracy (BoW):", accuracy_score(y_test, preds_bow))

    # 4. Load GloVe embeddings
    print("\nLoading GloVe...")
    embeddings = load_glove(glove_path, word2id, dim=embedding_dim)

    # 5. Average embeddings for each text
    print("\nEmbeddings-based features:")
    X_train_emb = np.array([text_to_vector(t, word2id, embeddings) for t in X_train])
    X_test_emb = np.array([text_to_vector(t, word2id, embeddings) for t in X_test])
    clf_emb = LogisticRegression(max_iter=200)
    clf_emb.fit(X_train_emb, y_train)
    preds_emb = clf_emb.predict(X_test_emb)
    print("Test Accuracy (with GloVe embeddings):", accuracy_score(y_test, preds_emb))

    # 6. Compare
    print("\nSummary:")
    print(f"Bag-of-words accuracy: {accuracy_score(y_test, preds_bow):.4f}")
    print(f"GloVe embeddings accuracy: {accuracy_score(y_test, preds_emb):.4f}")

if __name__ == "__main__":
    main()