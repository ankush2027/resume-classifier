import pandas as pd

df = pd.read_csv("data/resume_dataset.csv")

#first 5 rows
print(df.head())

print(df.info())

#number of resumes per category
print(df['Category'].value_counts())