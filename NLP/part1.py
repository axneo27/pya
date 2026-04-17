import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Завантаження необхідних мовних ресурсів
# nltk.download("punkt")  # Для токенізації
# nltk.download("stopwords")  # Стоп-слова
# nltk.download("wordnet")  # Для лемматизації
# nltk.download("omw-1.4")  # WordNet мовні дані

# Вхідний текст
text = "Cats are chasing mice. The weather was beautiful, and we went out to play!"
# text = "Too small, would prefer a mat for a work surface that was anti static."

# --- 1. Токенізація ---
print("=== Токенізація ===")
sentences = sent_tokenize(text)
print("Речення:", sentences)

words = word_tokenize(text)
print("Слова:", words)

# --- 2. Стоп-слова ---
print("\n=== Видалення стоп-слів ===")
stop_words = set(stopwords.words("english"))
filtered_words = [
    word for word in words if word.lower() not in stop_words and word.isalpha()
]
print("Без стоп-слів:", filtered_words)

# --- 3. Стеммінг ---
print("\n=== Стеммінг ===")
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("Після стеммінгу:", stemmed_words)

# --- 4. Лемматизація ---
print("\n=== Лемматизація ===")
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in filtered_words]
print("Після лемматизації:", lemmatized_words)

nltk.download()
