
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

if selected_items:
    st.write(f"You selected: {selected_items}")

    # Parameters for Apriori
    min_support = st.slider("Minimum Support", 0.1, 1.0, 0.2)
    min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.6)

    # Apply Apriori Algorithm
    frequent_itemsets = apriori(df_transformed, min_support=min_support, use_colnames=True)
    num_i = len(frequent_itemsets)
    if not frequent_itemsets.empty:
        st.subheader("Frequent Itemsets")
        st.write(frequent_itemsets)

        # Generate Association Rules
        rules = association_rules(frequent_itemsets, num_itemsets=num_i, metric="confidence", min_threshold=min_confidence)

        if not rules.empty:
            st.subheader("All Association Rules")
            st.write(rules)

            # Filter rules based on selected items
            filtered_rules = rules[
                rules['antecedents'].apply(lambda x: any(item in x for item in selected_items)) |
                rules['consequents'].apply(lambda x: any(item in x for item in selected_items))
            ]

            st.subheader("Filtered Rules Based on Selected Items")
            if not filtered_rules.empty:
                st.write(filtered_rules)
            else:
                st.write("No rules found for the selected item(s).")
        else:
            st.write("No association rules found. Try lowering the minimum confidence.")
    else:
        st.write("No frequent itemsets found. Try lowering the minimum support.")
else:
    st.write("Please select at least one item to start the analysis.")