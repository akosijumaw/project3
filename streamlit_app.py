import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

st.title('🎈 Jumar Buladaco Project')

st.write('Hello world!')
file_path = 'https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/GroceryStoreDataSet.csv' 
data = pd.read_csv(file_path, header=None)
data
