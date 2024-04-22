import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_flat = pd.read_csv('./webui/df_flat.csv')
print('-----------------------------------')

def region_analytics(df_flat, target_type, target):
    
    target = str(target)
    target_type = str(target_type)
    print(target)

    unique_id = df_flat[df_flat[target_type] == target].id.unique()
    print(unique_id)
    
    tmp = df_flat[(df_flat['year'].isin([2019, 2020, 2021, 2022, 2024])) & (df_flat['id'].isin(unique_id))].groupby(['year', 'rooms_en']).actual_worth.mean()
    tmp = pd.DataFrame(tmp).reset_index()
    print('this is result tmp: ', tmp)
#    tmp = st.dataframe(tmp)

    sns.lineplot(x='year', y='actual_worth', hue='rooms_en', data=tmp)

target_type = st.sidebar.text_input('Choose Target Type', 'nearest_mall_en')
target = st.sidebar.text_input('Choose Target Name', 'Marina Mall')

st.title(f'Mean prices for estates near given object: \n {target_type}  {target}')

st.sidebar.subheader('Region Analytics')



fig = plt.figure(figsize=(10,4))

region_analytics(df_flat = df_flat,
                            target_type = target_type, 
                            target = target)

st.pyplot(fig)
