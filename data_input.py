# data_input.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataInput:
    def __init__(self, filepath):
        """
        Initializes the class with the provided CSV file path and loads the data into a pandas DataFrame.

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

    def scale_data(self):
        """
        Standardize the data (scale it).
        """
        # Standardizing the exam marks data (Q1 to Q5)
        scaler = StandardScaler()
        scaled_df = self.df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
        scaled_df = pd.DataFrame(scaler.fit_transform(scaled_df), columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        # Return the scaled data along with other necessary columns (e.g., Programme)
        self.df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']] = scaled_df
        return self.df

    def get_data(self):
        """
        Return the cleaned and scaled data.
        """
        return self.df
