import pandas as pd
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Dataset
try:
    df = pd.read_csv("extended_spam_dataset.csv")
    if len(df.columns) >= 2:
        df = df.iloc[:, :2]
        df.columns = ["label", "text"]
    else:
        print("Error: Dataset does not have enough columns.")
        exit()
except FileNotFoundError:
    print("Error: extended_spam_dataset.csv not found.")
    exit()

# Filter valid labels if needed
df = df[df["label"].isin(["ham", "spam"])]

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure stopwords and wordnet are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))
somali_stopwords = {
    "iyo", "waa", "in", "ka", "ku", "si", "ayaa", "ma", "haa", "leh", "loo",
    "la", "u", "wax", "badan", "ahay", "karo", "mid", "kuma", "wuu", "waxa"
}
all_stopwords = english_stopwords.union(somali_stopwords)

# Clean Text Function (Must match app.py)
def clean_text(text):
    # 1. Convert to lowercase
    text = str(text).lower()
    # 2. Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Tokenize
    words = text.split()
    # 4. Lemmatization + Stopword removal
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in all_stopwords]
    # 5. Join words back to text
    return " ".join(cleaned_words)

df['cleaned_text'] = df['text'].apply(clean_text)

# Vectorization and Model Training
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

# Improved Vectorizer settings
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=6000)
X_train_tfidf = vectorizer.fit_transform(X_train)

print("Training SVM...")
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_tfidf, y_train)

# SVM Evaluation
X_test_tfidf = vectorizer.transform(X_test)
y_pred_svm = svm_model.predict(X_test_tfidf)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")

# Save Models
print("Saving SVM model to spam_app/ directory...")
with open('spam_app/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('spam_app/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("SVM model and vectorizer saved successfully.")
