# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Завантаження необхідних мовних ресурсів
nltk.download("punkt")  # Для токенізації
nltk.download("punkt_tab")  # Для токенізації речень
nltk.download("stopwords")  # Стоп-слова
nltk.download("wordnet")  # Для лемматизації
nltk.download("omw-1.4")  # WordNet мовні дані

warnings.filterwarnings(action="ignore")


#  Reads ‘alice.txt’ file
sample = open("./text.txt", "r", encoding="utf-8")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# --- 2. Стоп-слова ---
print("\n=== Видалення стоп-слів ===")
stop_words = set(stopwords.words("english"))
filtered_words = [
    word for sentence in data for word in sentence if word.lower() not in stop_words and word.isalpha()
]

# Create CBOW model
model1 = gensim.models.Word2Vec([filtered_words], min_count=1, vector_size=100, window=5)

print(model1.wv.most_similar("children", topn=5))

# Print results
print(
    "Cosine similarity between 'alice' " + "and 'wonderland' - CBOW : ",
    model1.wv.similarity("alice", "wonderland"),
)

print(
    "Cosine similarity between 'alice' " + "and 'machines' - CBOW : ",
    model1.wv.similarity("alice", "machines"),
)

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)

# Print results
print(
    "Cosine similarity between 'alice' " + "and 'wonderland' - Skip Gram : ",
    model2.wv.similarity("alice", "wonderland"),
)

print(
    "Cosine similarity between 'alice' " + "and 'machines' - Skip Gram : ",
    model2.wv.similarity("alice", "machines"),
)
