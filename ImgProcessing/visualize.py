import plotly.express as px
import pandas as pd 

df = pd.read_csv('../Labelling/original_labels.csv')
df = df[['fabric_group_name']]
df.rename(columns={'fabric_group_name': 'fabric'}, inplace=True)
# df = df.groupby('fabric').count().reset_index()
# print(df.head())
fig = px.bar(df, x='fabric')
fig.show()

# import pandas as pd 
# import os 
# import plotly.express as px
# data = {'class_code':[],
#          'count': []}

# path = '/userhome/2072/fyp22007/data/processed_images/'

# for dirpath, dirnames, files in os.walk(path):
#     dirname = dirpath.split(os.path.sep)[-1]
#     if dirname != '':
#         data['class_code'].append(dirname)
#         data['count'].append(len(files))
# df = pd.DataFrame(data)
# df['class_code'] = df['class_code'].astype(int)
# df.sort_values(by=['class_code'], inplace=True)
# print(df.head())

# fig = px.bar(df, x='class_code', y='count')
# fig.write_image("/userhome/2072/fyp22007/MLinAraechology/test.jpg")