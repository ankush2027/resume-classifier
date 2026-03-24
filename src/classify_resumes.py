import pandas as pd
import pickle
import re
import os

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
with open("models/model.pkl", "rb") as f:
    model, tfidf = pickle.load(f)

print("Reading resumes...")

# read CSV file containing resumes
df = pd.read_csv("data/input/resumes_to_classify.csv")

# clean resumes
df["cleaned_resume"] = df["Resume"].apply(clean_resume)

print("Converting resumes to TF-IDF vectors...")

X = tfidf.transform(df["cleaned_resume"])

print("Predicting job categories...")

predictions = model.predict(X)

# add predictions to dataframe
df["Predicted_Category"] = predictions

# remove cleaned column (not needed for HR)
df = df.drop(columns=["cleaned_resume"])

# create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

print("Saving grouped resume files...")

# group resumes by predicted category
for category in df["Predicted_Category"].unique():

    category_df = df[df["Predicted_Category"] == category]

    filename = category.replace(" ", "_") + "_resumes.csv"

    filepath = os.path.join("output", filename)

    category_df.to_csv(filepath, index=False)

    print(f"Saved {filepath}")

print("All resumes classified successfully!")