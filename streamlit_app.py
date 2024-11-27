import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

st.title('🎈 Jumar Buladaco Project')

st.write('Hello world!')
file_path = 'https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/GroceryStoreDataSet.csv' 
data = pd.read_csv(file_path, header=None)
transactions = data.values.tolist()


# Transform the list of transactions into a one-hot-encoded dataframe
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df_transformed = pd.DataFrame(te_array, columns=te.columns_)


# Display the one-hot-encoded dataframe
df_transformed


# Find frequent itemsets with a minimum support value
frequent_itemsets = apriori(df_transformed, min_support=0.2, use_colnames=True)

# Display the frequent itemsets
frequent_itemsets

# Generate association rules with a minimum confidence value
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Display the rules
rules