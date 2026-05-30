from gensim.models import Word2Vec
import gensim.downloader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from collections import Counter

corpus = [
    "This product is absolutely amazing and I love the quality.",
    "The quality of this brand is really good and worth buying.",
    "Bad experience with the service, the product was terrible.",
    "Great brand, great service, great quality overall.",
    "I had a poor experience, the item was awful and cheap.",
    "Amazing product, very nice design and good packaging.",
    "The brand is excellent, customer service is outstanding.",
    "This was a bad purchase, quality is poor and delivery was slow.",
    "I love this product, it works great and looks nice.",
    "Terrible quality, I would not recommend this brand at all.",
    "Good value for money, nice product with great features.",
    "The service was awful, product broke after one day of use.",
    "Outstanding quality and amazing design, highly recommend.",
    "Very nice brand, good prices and excellent customer support.",
    "Poor packaging but the product itself is good and reliable.",
]

# Task 1
def preprocess(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()

tokenized = [preprocess(s) for s in corpus]

print(f"Sentences: {len(tokenized)}")
print(f"Avg length: {sum(len(s) for s in tokenized) / len(tokenized):.2f} words")

# Task 2
model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, sg=1, epochs=20)

if "good" in model.wv:
    print("\nVector for 'good':\n", model.wv["good"])

# Task 3
print("\n{:<15} {:<30} {:<10}".format("Target", "Similar words", "Similarity"))
print("-" * 60)
for word in ["good", "bad", "quality"]:
    if word in model.wv:
        similar = model.wv.most_similar(word, topn=3)
        words = ", ".join(w for w, _ in similar)
        scores = ", ".join(f"{s:.2f}" for _, s in similar)
        print(f"{word:<15} {words:<30} {scores}")

# Task 4
all_words = [w for sent in tokenized for w in sent]
top20 = [w for w, _ in Counter(all_words).most_common(20) if w in model.wv]

vectors = [model.wv[w] for w in top20]
pca = PCA(n_components=2)
reduced = pca.fit_transform(vectors) # type: ignore

plt.figure(figsize=(10, 8))
for i, word in enumerate(top20):
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x, y, word, fontsize=10)
plt.title("Word2Vec — PCA 2D")
plt.grid(True)
plt.tight_layout()
plt.savefig("NLP/word2vec_visualization.png")
plt.show()

# GloVe comparison
glove = gensim.downloader.load("glove-wiki-gigaword-100")

pairs = [("good", "great"), ("bad", "terrible"), ("quality", "excellent"), ("amazing", "nice")]
print("\n{:<20} {:<15} {:<15}".format("Pair", "Word2Vec", "GloVe"))
print("-" * 50)
for w1, w2 in pairs:
    w2v_sim = model.wv.similarity(w1, w2) if w1 in model.wv and w2 in model.wv else None
    glove_sim = glove.similarity(w1, w2) if w1 in glove and w2 in glove else None # type: ignore
    w2v_str = f"{w2v_sim:.4f}" if w2v_sim is not None else "N/A"
    glove_str = f"{glove_sim:.4f}" if glove_sim is not None else "N/A"
    print(f"{w1}-{w2:<15} {w2v_str:<15} {glove_str}")
