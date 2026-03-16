import pandas as pd
import pickle
import re

# text cleaning function
def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


print("Loading trained model...")

# load model and vectorizer
with open("model.pkl", "rb") as f:
    model, tfidf = pickle.load(f)

print("Reading resumes...")

# read CSV file containing resumes
df = pd.read_csv("resumes_to_classify.csv")

# clean resumes
df["cleaned_resume"] = df["Resume"].apply(clean_resume)

print("Converting resumes to TF-IDF vectors...")

X = tfidf.transform(df["cleaned_resume"])

print("Predicting job categories...")

predictions = model.predict(X)

# add predictions to dataframe
df["Predicted_Category"] = predictions
df = df.drop(columns=["cleaned_resume"])

print("Saving results...")

df.to_csv("classified_resumes.csv", index=False)

print("Done! File saved as classified_resumes.csv")