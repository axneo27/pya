from textblob import TextBlob

text1 = "I love this product. It's amazing!"
text2 = "This is the worst experience I've ever had."

blob1 = TextBlob(text1)
blob2 = TextBlob(text2)

print(f"Text 1: {text1}")
print("Sentiment:", blob1.sentiment)  # -1 ... 1

print("\nText 2:", text2)
print("Sentiment:", blob2.sentiment)
