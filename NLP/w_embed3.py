from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re

# 📌 1. Підготовка корпусу
corpus = [
    "Король живе в палаці.",
    "Королева є його дружиною.",
    "Чоловік і жінка – дві статі.",
    "Кішка ловить мишу.",
    "Собака охороняє дім.",
    "Лев є царем звірів.",
    "Тигр також сильний хижак.",
]


# Очистка і токенізація
def preprocess(text):
    text = re.sub(
        r"[^\w\s]", "", text.lower()
    )  # нижній регістр, видалення розділових знаків
    return text.split()


tokenized_corpus = [preprocess(sentence) for sentence in corpus]

# 📌 2. Побудова моделі Word2Vec
model = Word2Vec(
    sentences=tokenized_corpus, vector_size=100, window=3, min_count=1, workers=4
)
model.save("model_w2v.model")

# 📌 3. Аналіз
print("\n🔍 Схожі до 'жінка':")
print(model.wv.most_similar("жінка", topn=5))

print("\n🧮 Семантична арифметика (король - чоловік + жінка):")
result = model.wv.most_similar(
    positive=["король", "жінка"], negative=["чоловік"], topn=3
)
print(result)


# 4. Візуалізація схожих слів (TSNE)
def visualize(model, target_word, topn=10):
    words = [target_word]
    embeddings = [model.wv[target_word]]

    for word, _ in model.wv.most_similar(target_word, topn=topn):
        words.append(word)
        embeddings.append(model.wv[word])

    tsne = TSNE(n_components=2, random_state=0, perplexity=5)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=12)
    plt.title(f"Схожі слова до '{target_word}' (TSNE)")
    plt.grid(True)
    plt.show()


visualize(model, "жінка")
