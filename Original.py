# ------------------------------
# ORIGINAL APRIORI CODE
# ------------------------------

# Install required libraries:
# pip install pandas mlxtend matplotlib

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# ------------------------------
# STEP 1: Load the dataset
# ------------------------------

dataset = [
    ['Milk', 'Bread', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Eggs'],
    ['Milk', 'Eggs'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

# Convert to One-Hot DataFrame
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)

print("Dataset After One-Hot Encoding:")
print(df)

# ------------------------------
# STEP 2: Run Apriori
# ------------------------------

frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# ------------------------------
# STEP 3: Generate Association Rules
# ------------------------------

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print("\nAssociation Rules:")
print(rules)

# ------------------------------
# STEP 4: Plot Support vs Confidence
# ------------------------------

plt.scatter(rules['support'], rules['confidence'])
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence (Original Apriori)")
plt.show()
