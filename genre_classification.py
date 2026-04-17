import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

texts = []
labels = []
with open('train_data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(':::')

        if len(parts) >= 3:
            texts.append(parts[1].strip())   
            labels.append(parts[2].strip()) 
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

sample = ["A love story between two people"]
sample_vec = vectorizer.transform(sample)
print("Predicted Genre:", model.predict(sample_vec))