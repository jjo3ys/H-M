import numpy as np
import pandas as pd
import csv

def cos_sim(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

def min_max(df):
    return (df-df.min())/(df.max()-df.min())
    
df = pd.read_csv('articles_only_num.csv')
header = df.iloc[:, 1].to_numpy().reshape(len(df))
df = df.iloc[:, 2:]
df['graphical_appearance_no'] -= 1010000
df['garment_group_no'] -= 1000
df['product_code'] = min_max(df["product_code"])
df['product_type_no'] = min_max(df['product_type_no'])
df['department_no'] = min_max(df['department_no'])
df['garment_group_no'] = min_max(df['garment_group_no'])
print(df)
df = df.to_numpy()

with open("cosine_similarity.csv", "w", encoding="utf-8", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(header)
    for i in range(len(df)):
        print(i)
        similarity = [None for i in range(len(df)+1)]
        similarity[0] = header[i]
        for j in range(i, len(df)):
            if i == j:
                similarity[j+1] = 1
            else:
                similarity[j+1] = cos_sim(df[i], df[j])

        wr.writerow(similarity)