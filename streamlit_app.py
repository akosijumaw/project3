import streamlit as st
import pandas as pd

st.title('🎈 Jumar Buladaco Project')

st.write('Hello world!')
file_path = 'https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/Processed_GroceryStoreDataSet.csv' 
data = pd.read_csv(file_path, header=None)
transactions = data.values.tolist()

from mlxtend.preprocessing import TransactionEncoder

# Transform the list of transactions into a one-hot-encoded dataframe
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_array, columns=te.columns_)

# Display the one-hot-encoded dataframe
print(df_transformed.head())