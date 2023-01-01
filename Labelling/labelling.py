# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def generateEncoding(): 
    # Import dataset
    df = pd.read_csv('../Labelling/original_labels.csv')
    df = df[['file_name', 'fabric_group_name']]
    df.rename(columns={'fabric_group_name': 'fabric'}, inplace=True)
    # verify integrity
    if df.file_name.nunique != df.file_name.nunique:
        print("There are duplicated labels")

    else:
        df = df.dropna()  # remove rows with NaN values

        # Label Encoding
        df_labelEncoding = df.copy()
        le = LabelEncoder()
        df_labelEncoding.fabric = le.fit_transform(
            df_labelEncoding.fabric.astype(str))
        filepath = Path('../Labelling/labelEncoding.csv')
        df_labelEncoding.to_csv(filepath)

        le_name_mapping = pd.DataFrame(
            zip(le.classes_, le.transform(le.classes_)), columns=['Class', 'EncodedLabel'])
        filepath = Path('../Labelling/labelEncodingMapping.csv')
        le_name_mapping.to_csv(filepath)

        # One-Hot Encoding
        one_hot_encoded_data = pd.get_dummies(df, columns=['fabric'])
        filepath = Path('../Labelling/oneHotEncoding.csv')
        one_hot_encoded_data.to_csv(filepath)
        
if __name__ == '__main__':
    generateEncoding()
