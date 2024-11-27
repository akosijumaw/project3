import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Title and Description
st.title("Apriori Association Rules Mining")
st.write("by Jumar S. Buladaco")

# Step 1: File Upload
uploaded_file = st.file_uploader("", type="csv")

data = pd.read_csv(uploaded_file, header=None)  
# Assuming no headers in the file
st.write("Uploaded Dataset:")
st.write(data)

# Step 3: Preprocess the data into transactions
transactions = data[0].apply(lambda x: x.split(',')).tolist()

# Transaction Encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_transformed = pd.DataFrame(te_array, columns=te.columns_)

st.subheader("Dataset (Preprocessed Transactions)")
st.write(df_transformed)

# Step 4: Extract unique items for user selection
all_items = sorted(te.columns_)

# Let the user select items
selected_items = st.multiselect("Select an item or items to include in the analysis:", options=all_items)


        

if selected_items:
    st.write(f"You selected: {selected_items}")

    # Filter dataset to include only transactions containing the selected items
    filtered_transactions = [
        transaction for transaction in transactions if any(item in transaction for item in selected_items)
        ]

    #st.sheader("Filtered Transactions")
    #st.write(filtered_transactions)

    # Encode the filtered transactions
    te_filtered = TransactionEncoder()
    te_filtered_array = te_filtered.fit(filtered_transactions).transform(filtered_transactions)
    df_filtered = pd.DataFrame(te_filtered_array, columns=te_filtered.columns_)

    # Step 5: Set Parameters for Apriori
    min_support = st.slider("Minimum Support", 0.1, 1.0, 0.2)
    min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.6)

    # Apply Apriori Algorithm
    frequent_itemsets = apriori(df_filtered, min_support=min_support, use_colnames=True)
    num_i = len(frequent_itemsets)

    if not frequent_itemsets.empty:
        st.subheader("Frequent Itemsets (Filtered)")
        st.write(frequent_itemsets)

                # Generate Association Rules
        rules = association_rules(frequent_itemsets, num_itemsets=num_i, metric="confidence", min_threshold=min_confidence)

        if not rules.empty:
            st.subheader("Association Rules (Filtered)")
            st.write(rules)

            # Highlight rules involving the selected items
            relevant_rules = rules[
                rules['antecedents'].apply(lambda x: any(item in x for item in selected_items)) |
                rules['consequents'].apply(lambda x: any(item in x for item in selected_items))
                ]

            st.subheader("Rules Involving Selected Items")
            if not relevant_rules.empty:
                st.write(relevant_rules)
            else:
                st.write("No rules found involving the selected item(s).")
        else:
            st.write("No association rules found. Try lowering the minimum confidence.")
    else:
        st.write("No frequent itemsets found. Try lowering the minimum support.")
else:
            st.write("Please select at least one item to start the analysis.")



