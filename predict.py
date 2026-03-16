import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# load dataset to rebuild TF-IDF
df = pd.read_csv("data/resume_dataset.csv")

def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# rebuild TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
df['cleaned'] = df['Resume'].apply(clean_resume)

tfidf.fit(df['cleaned'])

# load saved model
model = pickle.load(open("model.pkl", "rb"))

print("Model loaded successfully")


# user input for multiple resume prediction
print("\nResume Role Classifier Started")
print("Paste resume text (type 'exit' to stop)\n")

while True:

    sample_resume = input("Enter resume text:\n")

    cleaned = clean_resume(sample_resume)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)

    print("Predicted Category:", prediction[0])
    
    if sample_resume.lower() == "exit":
        print("Exiting classifier...")
        break

