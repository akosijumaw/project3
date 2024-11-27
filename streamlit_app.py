import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Title and Description
st.title("Apriori Association Rules Mining")
st.write("Explore frequent itemsets and association rules using a preloaded dataset.")

df = pd.read_csv('https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/Market_Basket_Optimisation%20-%20Market_Basket_Optimisation.csv', header=None)
transactions = df.values.tolist()
df

items = set()
for col in df:
    items.update(df[col].unique())
items


#Data Preprocessing
itemset = set(items)
encoded_vals = []
for index, row in df.iterrows():
    rowset = set(row)
    labels = {}
    uncommons = list(itemset - rowset)
    commons = list(itemset.intersection(rowset))
    for uc in uncommons:
        labels[uc] = 0
    for com in commons:
        labels[com] = 1
    encoded_vals.append(labels)
encoded_vals[0]
ohe_df = pd.DataFrame(encoded_vals)

