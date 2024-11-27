import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# Title and Description
st.title("Apriori Association Rules Mining")
st.write("by Jumar Buladaco")


#file_path = 'https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/Market_Basket_Optimisation%20-%20Market_Basket_Optimisation.csv' 
#data = pd.read_csv(file_path, header=None)
#df.head(10)

df = pd.read_csv('https://gist.githubusercontent.com/Harsh-Git-Hub/2979ec48043928ad9033d8469928e751/raw/72de943e040b8bd0d087624b154d41b2ba9d9b60/retail_dataset.csv', sep=',')
df

items = set()
for col in df:
    items.update(df[col].unique())
items

