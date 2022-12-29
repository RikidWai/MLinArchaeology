import os
import pandas as pd

rootdir = '/content/drive/MyDrive/export_images'
targetdir = '/content/drive/MyDrive/CS_FYP_archaeology/test_images/'

df_encoding = pd.read_csv('LabelEncoding.csv')

# Loop over differ
for root, dirs, files in os.walk(rootdir):
    for file in files:
        filename, extension = os.path.splitext(file)
        if 'cr' in extension or 'CR' in extension:
            path = os.path.join(root, file)
            dir = root.split(os.path.sep)[-1]
            print(f'{dir}_{filename}_s{1}')

            # check the dir match filename column in LabelEncoding, then put into respective folder
            targetFolder = str(df_encoding.query(f'file_name == "{dir}"').fabric)
            print(targetdir + targetFolder + os.path.sep)
            if not os.path.exists(targetFolder):
                os.makedirs(targetFolder)

