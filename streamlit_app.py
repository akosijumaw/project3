import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Title and Description
st.title("Apriori Association Rules Mining")
st.write("Explore frequent itemsets and association rules using a preloaded dataset.")

# Preload the dataset
data = pd.DataFrame({
    0: [
        "MILK,BREAD,BISCUIT",
        "BREAD,MILK,BISCUIT,CORNFLAKES",
        "BREAD,TEA,BOURNVITA",
        "JAM,MAGGI,BREAD,MILK",
        "MAGGI,TEA,BISCUIT",
    ]
})

# Preprocess the data
transactions = data[0].apply(lambda x: x.split(',')).tolist()

# Transaction Encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_array, columns=te.columns_)

# Display the dataset
st.subheader("Dataset (Preprocessed Transactions)")
st.write(df_transformed)

# Parameters for Apriori
min_support = st.slider("Minimum Support", 0.1, 1.0, 0.2)
min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.6)

# Apply Apriori Algorithm
frequent_itemsets = apriori(df_transformed, min_support=min_support, use_colnames=True)
st.subheader("Frequent Itemsets")
st.write(frequent_itemsets)

# Generate Association Rules

st.subheader("Association Rules")
