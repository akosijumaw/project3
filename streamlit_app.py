
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




# Preprocess the data
transactions = df[0].apply(lambda x: x.split(',')).tolist()

# Transaction Encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_array, columns=te.columns_)

# Extract unique items for user selection
all_items = sorted(te.columns_)

# Step 1: Let the user select items
selected_items = st.multiselect("Select an item or items to analyze:", options=all_items)
