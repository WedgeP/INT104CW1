# student_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from data_input import DataInput  # Import the DataInput class

class StudentDataAnalysis:
    def __init__(self, filepath):
        """
        Initializes the class with the provided CSV file path and sets up the data.
        :param filepath: The file path of the CSV containing the student data.
        """
        # Initialize the DataInput class
        data_input = DataInput(filepath)
        data_input.clean_data()
        self.df = data_input.get_data()

    def pca_analysis(self):
        """
        Perform PCA on the exam marks data, determine the optimal number of components,
        and visualize the explained variance ratio.
        """
        # Standardize the data
        df_scaled = self.df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Total']]
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Total'])

        # Apply PCA and calculate explained variance ratio
        pca = PCA()
        pca.fit(df_scaled)

        # Explained variance ratio for each component
        explained_variance_ratio = pca.explained_variance_ratio_

        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--')
        plt.title('Explained Variance Ratio for Each PCA Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.show()

        # Plot cumulative explained variance
        cumulative_variance = explained_variance_ratio.cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='b')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.axhline(y=0.9, color='r', linestyle='--')  # 90% threshold line
        plt.show()

        # Print the cumulative variance to help select the number of components
        print(f"Cumulative explained variance: {cumulative_variance}")

        # You can select the number of components that explain at least 90% of the variance
        optimal_components = next(i for i, total_variance in enumerate(cumulative_variance) if total_variance >= 0.9)
        print(f"Optimal number of components: {optimal_components + 1}")

        return optimal_components + 1  # Return the optimal number of components
    def tsne_analysis(self):
        """
        Perform t-SNE analysis for visualization of high-dimensional data.
        """
        # Standardize the data (already done in DataInput)
        df_scaled = self.df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5','Total']]

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_components = tsne.fit_transform(df_scaled)

        # Create a DataFrame for t-SNE components
        tsne_df = pd.DataFrame(tsne_components, columns=['TSNE1', 'TSNE2'])
        tsne_df['Programme'] = self.df['Programme']

        # Visualize t-SNE result
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=tsne_df, x='TSNE1', y='TSNE2', hue='Programme', palette='Set1')
        plt.title('t-SNE of Exam Marks (Q1-Q5)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()

# Usage example:
if __name__ == "__main__":
    # Specify the file path
    filepath = 'student_data.csv'

    # Create an instance of StudentDataAnalysis
    analysis = StudentDataAnalysis(filepath)

    # Perform PCA and visualize
    analysis.pca_analysis()

    # Perform t-SNE and visualize
    analysis.tsne_analysis()
