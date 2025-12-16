import pandas as pd
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load Dataset
try:
    df = pd.read_csv("extended_spam_dataset.csv")
    # Ensure correct columns if they differ, but based on notebook they are 'label', 'message'
    # Notebook did: df.columns = ["label", "message"]
    # So let's align with that just in case
    if len(df.columns) >= 2:
        df = df.iloc[:, :2]
        df.columns = ["label", "text"] # Using 'text' to match my script's convention below
    else:
        print("Error: Dataset does not have enough columns.")
        exit()
except FileNotFoundError:
    print("Error: extended_spam_dataset.csv not found.")
    exit()

# Filter valid labels if needed (notebook did this)
df = df[df["label"].isin(["ham", "spam"])]

# Clean Text Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    # Simple stopword removal (English + Somali common words)
    stopwords = set(['the', 'is', 'in', 'and', 'to', 'of', 'a', 'for', 'on', 'with', 
                     'waa', 'iyo', 'ee', 'oo', 'ku', 'ka', 'u', 'si', 'aad'])
    text = ' '.join(word for word in text.split() if word not in stopwords)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

# Vectorization and Model Training
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

# Improved Vectorizer settings from notebook
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=6000)
X_train_tfidf = vectorizer.fit_transform(X_train)

svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_tfidf, y_train)

# Evaluation
X_test_tfidf = vectorizer.transform(X_test)
y_pred = svm_model.predict(X_test_tfidf)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred)}")

# Train LightGBM
import lightgbm as lgb
y_train_lgb = y_train.map({'ham': 0, 'spam': 1})
y_test_lgb = y_test.map({'ham': 0, 'spam': 1})

lgb_train = lgb.Dataset(X_train_tfidf, label=y_train_lgb)
lgb_eval = lgb.Dataset(X_test_tfidf, label=y_test_lgb, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

print("Training LightGBM model...")
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                callbacks=[lgb.early_stopping(stopping_rounds=10)])

y_pred_lgb = gbm.predict(X_test_tfidf, num_iteration=gbm.best_iteration)
y_pred_lgb_binary = [1 if x >= 0.5 else 0 for x in y_pred_lgb]
print(f"LightGBM Accuracy: {accuracy_score(y_test_lgb, y_pred_lgb_binary)}")

# Save Models
with open('spam_app/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('spam_app/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

gbm.save_model('spam_app/lightgbm_model.txt')

print("Model and vectorizer saved successfully.")
