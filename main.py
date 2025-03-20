import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class StudentDataAnalysis:
    def __init__(self, filepath):
        """
        Initializes the class with the provided CSV file path.
        Loads the data into a pandas DataFrame.

        :param filepath: The file path of the CSV containing the student data.
        """
        self.df = pd.read_csv(filepath)

    def clean_data(self):
        """
        Clean the data by checking for missing values and ensuring the data types are correct.
        Also, correct any inconsistent entries in the 'gender' and 'grade' columns.
        """
        # Checking for missing values
        missing_data = self.df.isnull().sum()
        print("Missing Data:\n", missing_data)

        # Fill missing values if necessary (you can choose to fill or drop)
        self.df = self.df.dropna()

        # Ensuring the 'gender' and 'grade' columns are integers
        self.df['Gender'] = self.df['Gender'].astype(int)
        self.df['Grade'] = self.df['Grade'].astype(int)

        # Checking for inconsistent entries
        if not set(self.df['Gender']).issubset({1, 2}):
            print("Warning: Unexpected gender values detected.")
        if not set(self.df['Grade']).issubset({2, 3}):
            print("Warning: Unexpected grade values detected.")

    def basic_statistics(self):
        """
        Display basic descriptive statistics for the marks (Q1-Q5) and total marks.
        """
        print("\nBasic Descriptive Statistics:")
        print(self.df.describe())

    def performance_by_gender(self):
        """
        Compare the performance of male and female students based on their average marks.
        """
        gender_avg_marks = self.df.groupby('Gender')[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean()
        print("\nAverage Marks by Gender:")
        print(gender_avg_marks)

        # Plot the results
        gender_avg_marks.plot(kind='bar', figsize=(10, 6))
        plt.title('Average Marks by Gender')
        plt.ylabel('Average Marks')
        plt.show()

    def performance_by_programme(self):
        """
        Compare the performance of students across different programmes.
        """
        programme_avg_marks = self.df.groupby('Programme')[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean()
        print("\nAverage Marks by Programme:")
        print(programme_avg_marks)

        # Plot the results
        programme_avg_marks.plot(kind='bar', figsize=(10, 6))
        plt.title('Average Marks by Programme')
        plt.ylabel('Average Marks')
        plt.show()

    def performance_by_grade(self):
        """
        Compare the performance of students based on their grade level (Grade 2 or Grade 3).
        """
        grade_avg_marks = self.df.groupby('Grade')[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].mean()
        print("\nAverage Marks by Grade:")
        print(grade_avg_marks)

        # Plot the results
        grade_avg_marks.plot(kind='bar', figsize=(10, 6))
        plt.title('Average Marks by Grade')
        plt.ylabel('Average Marks')
        plt.show()

    def correlation_analysis(self):
        """
        Perform correlation analysis between individual exam questions and total marks.
        """
        correlation_matrix = self.df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].corr()
        print("\nCorrelation Matrix:")
        print(correlation_matrix)

        # Plot the heatmap of correlations
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title('Correlation Heatmap')
        plt.show()

    def mark_distribution(self):
        """
        Plot histograms for the distribution of marks in each exam question and total marks.
        """
        marks_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

        # Plot histograms
        self.df[marks_columns].hist(figsize=(12, 8), bins=20, edgecolor='black')
        plt.suptitle('Marks Distribution for Each Question and Total Marks')
        plt.show()

    def scaled_data_distribution(self):
        """
        Scales the data and visualizes the distribution after scaling.
        """
        # Scale the data (standardization)
        scaler = StandardScaler()
        df_scaled = self.df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].copy()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        # Add programme column back to the scaled data
        df_scaled['Programme'] = self.df['Programme']

        # Visualize the distribution after scaling
        plt.figure(figsize=(12, 8))
        for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
            plt.subplot(2, 3, i + 1)
            sns.boxplot(x='Programme', y=q, data=df_scaled)
            plt.title(f'Distribution of {q} by Programme (Scaled)')
        plt.tight_layout()
        plt.show()

    def pca_analysis(self):
        """
        Perform PCA (Principal Component Analysis) on the exam marks data and visualize the result.
        """
        # Standardize the data
        scaler = StandardScaler()
        df_scaled = self.df[['Gender','Grade','Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
        df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=['Gender','Grade','Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        # Apply PCA
        pca = PCA(n_components=3)
        pca_components = pca.fit_transform(df_scaled)

        # Create a DataFrame for PCA components
        pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2','PC3'])
        pca_df['Programme'] = self.df['Programme']

        # Visualize PCA result in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for the first 3 principal components
        scatter = ax.scatter(
            pca_df['PC1'], pca_df['PC2'], pca_df['PC3'],
            c=pca_df['Programme'].astype('category').cat.codes, cmap='Set1'
        )

        # Add labels
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('3D PCA of Exam Marks (Q1-Q5)')

        # Add a legend based on Programme
        handles, labels = scatter.legend_elements()
        ax.legend(handles, labels, title="Programme")

        # Show the plot
        plt.show()

    def clustering_analysis(self):
        """
        Perform K-Means clustering on the scaled data and evaluate clustering performance.
        """
        # Standardize the data
        scaler = StandardScaler()
        df_scaled = self.df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
        df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(df_scaled)

        # Evaluate clustering performance using silhouette score
        silhouette_avg = silhouette_score(df_scaled, self.df['Cluster'])
        print(f"Silhouette Score for K-Means clustering: {silhouette_avg}")

        # Visualize the clustering result
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Q1', y='Q2', hue='Cluster', data=self.df, palette='Set2')
        plt.title('K-Means Clustering of Students')
        plt.xlabel('Q1 Marks')
        plt.ylabel('Q2 Marks')
        plt.show()


# Usage
analysis = StudentDataAnalysis('student_data.csv')
analysis.clean_data()
analysis.basic_statistics()
analysis.performance_by_gender()
analysis.performance_by_programme()
analysis.performance_by_grade()
analysis.correlation_analysis()
analysis.mark_distribution()
analysis.scaled_data_distribution()
analysis.pca_analysis()
analysis.clustering_analysis()
