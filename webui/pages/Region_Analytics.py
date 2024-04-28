import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from st_pages import Page, Section, show_pages, add_page_title
import plotly.express as px

try:
    df_flat = pd.read_csv('./webui/df_flat.csv')
except:
    print('file not found')
try:
    df_flat = pd.read_csv('./df_flat.csv')
except:
    print('file not found')

print('-----------------------------------')

add_page_title()

def region_analytics(df_flat, target_type, target):
    
    target = str(target)
    target_type = str(target_type)
    print(target)

    unique_id = df_flat[df_flat[target_type] == target].id.unique()
    print(unique_id)
    
    tmp = df_flat[(df_flat['year'].isin([2019, 2020, 2021, 2022, 2024])) & (df_flat['id'].isin(unique_id))].groupby(['year', 'rooms_en']).actual_worth.mean()
    tmp = pd.DataFrame(tmp).reset_index()
    print('this is result tmp: ', tmp)
    return tmp


with st.container():
    target_type = st.selectbox(
    'Choose Target Type: ', ('nearest_mall_en', 'nearest_metro_en', 'nearest_landmark_en'))

    if target_type == 'nearest_mall_en':
        malls = tuple(df_flat.nearest_mall_en.unique())
        target = st.selectbox('Choose Mall: ', malls)
    
    elif target_type == 'nearest_metro_en':
        metro = tuple(df_flat.nearest_metro_en.unique())
        target = st.selectbox('Choose Metro: ', metro)

    elif target_type == 'nearest_landmark_en':
        landmark = tuple(df_flat.nearest_landmark_en.unique())
        target = st.selectbox('Choose Landmark: ', landmark)


st.title(f'Mean prices for estates near: ')
st.title(f'{target}')

st.sidebar.subheader('Region Analytics')




tmp = region_analytics(df_flat = df_flat,
                            target_type = target_type, 
                            target = target)

fig = px.line(tmp, x="year", y="actual_worth", color="rooms_en")


st.plotly_chart(fig)

