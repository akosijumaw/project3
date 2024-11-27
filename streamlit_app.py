
import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

# Title and Description
st.title("Apriori Association Rules Mining")
#st.write("Explore frequent itemsets and association rules using a preloaded dataset.")

data = pd.read_csv('https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/Market_Basket_Optimisation%20-%20Market_Basket_Optimisation.csv', header=None)

transactions = data.values.tolist()
    
# Transaction Encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_array, columns=te.columns_)

# Display the transformed data
st.subheader("One-Hot Encoded Transactions")
st.write(df_transformed.head())

# Parameters for Apriori
min_support = st.slider("Minimum Support", 0.1, 1.0, 0.2)
min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.6)

# Apply Apriori Algorithm
frequent_itemsets = apriori(df_transformed, min_support=min_support, use_colnames=True)
num_itemsets = len(frequent_itemsets)  # Get the number of frequent itemsets
st.subheader("Frequent Itemsets")
st.write(frequent_itemsets)

# Generate Association Rules
rules = association_rules(frequent_itemsets,num_itemsets=num_itemsets, metric="confidence", min_threshold=min_confidence)
st.subheader("Association Rules")
st.write(rules)