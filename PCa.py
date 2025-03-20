from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from main import StudentDataAnalysis


def pca_analysis(self):
    """
    Perform PCA (Principal Component Analysis) on the exam marks data and visualize the result.
    """
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = self.df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
    df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # Apply PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(df_scaled)

    # Check explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by each component: {explained_variance}")
    print(f"Total variance explained by 2 components: {explained_variance.sum()}")

    # Create a DataFrame for PCA components
    pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
    pca_df['Programme'] = self.df['Programme']

    # Visualize PCA result
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Programme', palette='Set1')
    plt.title('PCA of Exam Marks (Q1-Q5)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

# Assuming the method is within the StudentDataAnalysis class
analysis = pd.read_csv(filepath)
analysis.pca_analysis()
