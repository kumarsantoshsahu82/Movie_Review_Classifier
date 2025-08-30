# Installing dataset from Hugging Face and importing Libraries
!pip install datasets --quiet

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (25k reviews for training)
dataset = load_dataset("imdb")
df = pd.DataFrame(dataset['train'])

#  Train-test spliting for the dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

#  Convert text to TF-IDF features (Vectorizing text)
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

#  Training the Logistic Regression Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

#  Evaluating Model Performance
y_pred = model.predict(X_test_tfidf)
print(" Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))