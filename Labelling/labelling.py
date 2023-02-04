# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

def rmTypo():
    filepath = Path('../Labelling/object_finds_ceramics.csv')
    df = pd.read_csv(filepath)
    
    # Remove some typos 
    df.fabric_group_name = df.fabric_group_name.str.replace(r'\s+', ' ', regex=True) # Remove double spaces
    df.fabric_group_name = df.fabric_group_name.str.replace('rediish', 'reddish') # Remove double spaces
    df.fabric_group_name = df.fabric_group_name.str.replace('liight', 'light') # Remove double spaces
    df.fabric_group_name = df.fabric_group_name.str.replace('browm', 'brown') # Remove double spaces
    df.fabric_group_name = df.fabric_group_name.str.replace('medum', 'medium') # Remove double spaces
    df.fabric_group_name = df.fabric_group_name.str.lower()
    df.to_csv(filepath, index=False)

def encodingCol(df, col):
    le = LabelEncoder()
    df[col] = le.fit_transform(
        df[col].astype(str))
    le_name_mapping = pd.DataFrame(
        zip(le.classes_, le.transform(le.classes_)), columns=['Class', 'EncodedLabel'])
    filepath = Path(f'../Labelling/{col}LabelEncodingMapping.csv')
    le_name_mapping.to_csv(filepath)
    return df
    
    
def generateEncoding(): 
    # Import dataset
    filepath = Path('../Labelling/original_labels.csv')
    df = pd.read_csv(filepath)
    
   
    df = df[['file_name', 'fabric_group_name']]
    df.rename(columns={'fabric_group_name': 'fabric'}, inplace=True)
    # verify integrity
    if df.file_name.nunique != df.file_name.nunique:
        print("There are duplicated labels")

    else:
        df = df.dropna()  # remove rows with NaN values
        # df['fabric'] = df.fabric.str.rstrip('-1234567890.') # Remove -01, -02, ... in each labels 
        
        # print(df.head())
        # Label Encoding
        df_labelEncoding = df.copy()
        # df_labelEncoding['fabric'] = df_labelEncoding.fabric.str.rstrip('-1234567890.')
        df_labelEncoding[['color','texture']] = df_labelEncoding.fabric.str.extract('^(.*?)\s((?:dark|light)?\s?\S+)$') 
        
        
        df = encodingCol(df_labelEncoding, 'fabric')
        df = encodingCol(df_labelEncoding, 'color')
        df = encodingCol(df_labelEncoding, 'texture')
        
        filepath = Path('../Labelling/labelEncoding.csv')
        df_labelEncoding.to_csv(filepath)

        # One-Hot Encoding
        one_hot_encoded_data = pd.get_dummies(df, columns=['fabric'])
        filepath = Path('../Labelling/oneHotEncoding.csv')
        one_hot_encoded_data.to_csv(filepath)

def getNumClass(): 
    filepath = Path('../Labelling/labelEncodingMapping.csv', index = False)
    df = pd.read_csv(filepath)
    return len(df.index)

if __name__ == '__main__':
    rmTypo()
    generateEncoding()
    print(getNumClass())
