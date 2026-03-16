import pandas as pd
import re
import pickle

# Load dataset
df = pd.read_csv("data/resume_dataset.csv")

# First 5 rows
print(df.head())

print(df.info())

# Number of resumes per category
print(df['Category'].value_counts())


# Text cleaning function
def clean_resume(text):

    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text


# Apply cleaning
df['cleaned_resume'] = df['Resume'].apply(clean_resume)


# Convert text to numbers using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

X = tfidf.fit_transform(df['cleaned_resume'])
y = df['Category']

print("TF-IDF Shape:", X.shape)


# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train model using Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report (Naive Bayes):")
print(classification_report(y_test, y_pred))


# Train model using Logistic Regression
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_pred))


# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump((model, tfidf), f)

print("Model and TF-IDF vectorizer saved successfully")

