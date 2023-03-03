# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
directory = "/Users/mertfilizay/Desktop/Python_files/1000697904.parquet"

import pandas as pd
from sklearn.model_selection import train_test_split


# read the parquet file into a pandas DataFrame
df = pd.read_parquet(directory)

#Learn how many types there are in the dataset
#types = df['type'].unique()

#print(types)

# How many rows there are in the data?
#print(df.shape[0])  #23349


#check if there are missing values in the data
#print(df.isnull().sum())

#there are 1491 rows in the data without x,y,z coordinates.
#They have to be cleaned

df = df.dropna()

#print(df.isnull().sum())


# check for duplicates
#duplicates = df[df.duplicated()]

# print duplicates (if any)
#print(duplicates)  # no duplicates

Q1 = df[['x', 'y', 'z']].quantile(0.25)
Q3 = df[['x', 'y', 'z']].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[['x', 'y', 'z']] < (Q1 - 1.5 * IQR)) | (df[['x', 'y', 'z']] > (Q3 + 1.5 * IQR))).any(axis=1)

#prints out how many outliers there are in the data
#print(len(outliers[outliers==True])) #2597 outliers

#keeps only the non-outlier points in the dataset.
df = df[~outliers]


# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the training and validation sets as separate CSV files
train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)


