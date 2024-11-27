import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Title and Description
st.title("Apriori Association Rules Mining")
st.write("Explore frequent itemsets and association rules using a preloaded dataset.")

data = pd.read_csv('https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/Market_Basket_Optimisation%20-%20Market_Basket_Optimisation.csv', header=None)
transactions = data.values.tolist()
data

items = set()
for col in df:
    items.update(df[col].unique())
items