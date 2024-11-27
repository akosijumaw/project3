
import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

# Title and Description
st.title("Apriori Association Rules Mining")
st.write("by Jumar S. Buladaco")



with st.expander('Dataset'):
    df = pd.read_csv('https://gist.githubusercontent.com/Harsh-Git-Hub/2979ec48043928ad9033d8469928e751/raw/72de943e040b8bd0d087624b154d41b2ba9d9b60/retail_dataset.csv', header=None)
    transactions = df.values.tolist()
    df

items = set()
for col in df:
    items.update(df[col].unique())

#Data Preprocessing
with st.expander('Data Preprocessing'):
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
    #encoded_vals[0]
    ohe_df = pd.DataFrame(encoded_vals)
    ohe_df


with st.sidebar:
    st.header('Select')
    min_s = st.slider("Minimum Support", 0.1, 1.0, 0.2)
    min_c = st.slider("Minimum Confidence", 0.1, 1.0, 0.6)


# Parameters for Apriori


#Applying Apriori
freq_items = apriori(ohe_df, min_support=min_s, use_colnames=True, verbose=1)
num_i = len(freq_items)  # Get the number of frequent itemsets


#Mining Association Rules
rules = association_rules(freq_items, num_itemsets=num_i, metric='confidence', min_threshold=min_c)  # Pass num_itemsets

st.write("Association Rule")
rules


# Preprocess the data
transactions = df[0].apply(lambda x: x.split(',')).tolist()

# Transaction Encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_array, columns=te.columns_)