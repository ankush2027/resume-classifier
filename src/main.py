import pandas as pd

df = pd.read_csv("data/resume_dataset.csv")

#first 5 rows
print(df.head())

print(df.info())

#number of resumes per category
print(df['Category'].value_counts())


import re
#text cleaning function
import re

def clean_resume(text):

    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text

df['cleaned_resume'] = df['Resume'].apply(clean_resume)

#convert to numbers

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

X = tfidf.fit_transform(df['cleaned_resume'])
y = df['Category']

print(X.shape)



#Training the model

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#create model
model = MultinomialNB()

#train model
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))



#predicting category for a new resume
sample_resume = """
Experienced Python developer with knowledge of machine learning,
pandas, numpy, and data analysis.
"""

cleaned = clean_resume(sample_resume)

vector = tfidf.transform([cleaned])

prediction = model.predict(vector)

print("Predicted Category:", prediction[0])