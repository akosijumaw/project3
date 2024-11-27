import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules



file_path = 'https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/GroceryStoreDataSet.csv' 
data = pd.read_csv(file_path, header=None)

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
st.subheader("Frequent Itemsets")
st.write(frequent_itemsets)

    # Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
st.subheader("Association Rules")
st.write(rules)

