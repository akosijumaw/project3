import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules



file_path = 'https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/GroceryStoreDataSet.csv' 
data = pd.read_csv(file_path, header=None)




# Title and Description
st.title("Apriori Association Rules Mining")


# File Upload
uploaded_file = st.file_uploader("https://raw.githubusercontent.com/akosijumaw/data/refs/heads/main/GroceryStoreDataSet.csv", type=["csv"])

if uploaded_file:
    # Load the uploaded file
    data = pd.read_csv(uploaded_file, header=None)
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

    # Download Results
    if not frequent_itemsets.empty:
        st.download_button(
            label="Download Frequent Itemsets",
            data=frequent_itemsets.to_csv(index=False).encode("utf-8"),
            file_name="frequent_itemsets.csv",
            mime="text/csv",
        )

    if not rules.empty:
        st.download_button(
            label="Download Association Rules",
            data=rules.to_csv(index=False).encode("utf-8"),
            file_name="association_rules.csv",
            mime="text/csv",
        )