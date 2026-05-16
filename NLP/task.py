import json
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import matplotlib.pyplot as plt

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def process_review(text: str) -> dict:
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.isalpha() and w.lower() not in stop_words]
    stemmed = [stemmer.stem(w) for w in filtered]
    lemmatized = [lemmatizer.lemmatize(w.lower()) for w in filtered]
    return {
        "tokens": tokens,
        "filtered": filtered,
        "stemmed": stemmed,
        "lemmatized": lemmatized,
    }

def classify_sentiment(polarity: float) -> str:
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

reviews = []
with open('youtube-comments.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader, start=1):
        if row['Content'] and row['Content'].strip():
            reviews.append({
                "id": idx,
                "author": row['User'].replace('@', ''),
                "text": row['Content']
            })

results = []
for review in reviews:
    nlp = process_review(review["text"])
    results.append({
        "id": review["id"],
        "author": review["author"],
        "text": review["text"],
        "tokens": nlp["tokens"],
        "filtered": nlp["filtered"],
        "stemmed": nlp["stemmed"],
        "lemmatized": nlp["lemmatized"],
    })
    print(f"\n[{review['id']}] {review['author']}")
    print(f"  Текст:        {review['text']}")
    print(f"  Токени:       {nlp['tokens']}")
    print(f"  Без стоп-сл.: {nlp['filtered']}")
    print(f"  Стеммінг:     {nlp['stemmed']}")
    print(f"  Лемматиз.:    {nlp['lemmatized']}")

json_path = "reviews_processed.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nJSON збережено: {json_path}")

counts = {"positive": 0, "neutral": 0, "negative": 0}

for review in reviews:
    blob = TextBlob(review["text"])
    polarity = blob.sentiment.polarity # type: ignore
    sentiment = classify_sentiment(polarity)
    counts[sentiment] += 1
    print(f"[{review['id']}] {sentiment:>8}  (polarity={polarity:+.3f})  {review['text'][:55]}...")

print(f"\n  Позитивних : {counts['positive']}")
print(f"  Нейтральних: {counts['neutral']}")
print(f"  Негативних : {counts['negative']}")
print(f"  Всього     : {sum(counts.values())}")

sentiments = list(counts.keys())
values = list(counts.values())
colors = ['#2ecc71', '#f39c12', '#e74c3c']

plt.figure(figsize=(8, 6))
bars = plt.bar(sentiments, values, color=colors, edgecolor='black', linewidth=1.5)

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             str(value), ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.xlabel('Тональність', fontsize=12, fontweight='bold')
plt.ylabel('Кількість відгуків', fontsize=12, fontweight='bold')
plt.ylim(0, max(values) + 2)
plt.grid(axis='y', alpha=0.3, linestyle='--')

total = sum(values)
for i, (sentiment, value) in enumerate(zip(sentiments, values)):
    percentage = (value / total) * 100
    plt.text(i, value/2, f'{percentage:.1f}%', ha='center', va='center', 
             color='white', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()