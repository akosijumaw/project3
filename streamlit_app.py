import streamlit as st
import pandas as pd

st.title('🎈 Jumar Buladaco Project')

st.write('Hello world!')
file_path = 'https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/Processed_GroceryStoreDataSet.csv' 
data = pd.read_csv(file_path, header=None)
transactions = data.values.tolist()

data