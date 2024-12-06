# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the Iris dataset
from sklearn.datasets import load_iris

# Load the data into a pandas DataFrame
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target

# Mapping target numbers to species names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_map)

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Initial Plan for Data Exploration
# 1. Summary Statistics
print(df.describe())

# 2. Visualize the data distribution
sns.pairplot(df, hue="species")
plt.title("Pairplot of Iris Data Features by Species")
plt.show()

# 3. Checking for missing values
print("Missing values in each column:")
print(df.isnull().sum())

# Data Cleaning and Feature Engineering
# In this dataset, there are no missing values or major issues to handle.

# Key Findings and Insights

# 4. Correlation matrix to identify relationships
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Iris Features")
plt.show()

# Formulating Hypotheses
# Hypothesis 1: Petal length differs significantly across species.
# Hypothesis 2: Sepal width is correlated with species type.
# Hypothesis 3: Petal width is positively correlated with petal length.

# Hypothesis Testing: Formal significance test
# We can use ANOVA (Analysis of Variance) to check if the means of petal length differ significantly across species.

# Perform ANOVA for 'petal length' across different species
anova_result = stats.f_oneway(df[df['species'] == 'setosa']['petal length (cm)'],
                              df[df['species'] == 'versicolor']['petal length (cm)'],
                              df[df['species'] == 'virginica']['petal length (cm)'])

print(f"ANOVA result for petal length across species: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}")

# Based on the p-value, if it's below 0.05, we reject the null hypothesis (that the species have equal means for petal length)

# Summary of Key Insights:
# 1. There are clear differences in petal length between the species, as shown by the ANOVA test.
# 2. Petal width and petal length have a strong positive correlation, meaning they increase together.
# 3. The species are well-separated when comparing their petal-related features.

# Next Steps:
# Further analysis could involve applying supervised learning models (e.g., Decision Tree, Random Forest) to classify the species based on the features.
