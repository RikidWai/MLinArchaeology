import plotly.express as px
import pandas as pd 

df = pd.read_csv('../Labelling/original_labels.csv')
df = df[['fabric_group_name']]
df.rename(columns={'fabric_group_name': 'fabric'}, inplace=True)
# df = df.groupby('fabric').count().reset_index()
# print(df.head())
fig = px.bar(df, x='fabric')
fig.show()