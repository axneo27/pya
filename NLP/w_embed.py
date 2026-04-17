from gensim.models import Word2Vec

# Приклад корпусу
sentences = [
    ["король", "є", "чоловіком"],
    ["королева", "є", "жінкою"],
    ["чоловік", "і", "жінка"],
    ["парламент", "ухвалює", "закони"],
]

# Створення моделі Word2Vec
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, workers=4)

# Вектор для слова "король"
vector = model.wv["король"]
print("Вектор для слова 'король':\n", vector)

# Знайдемо слова, подібні до "король"
similar = model.wv.most_similar("король")
print("Схожі слова до 'король':\n", similar)
