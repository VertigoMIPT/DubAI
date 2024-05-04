import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from st_pages import Page, Section, show_pages, add_page_title
import plotly.express as px
from sklearn.preprocessing import normalize
import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost


class TargetEncode(BaseEstimator, TransformerMixin):
    
    def __init__(self, categories='auto', k=1, f=1, 
                 noise_level=0, random_state=None):
        if type(categories)==str and categories!='auto':
            self.categories = [categories]
        else:
            self.categories = categories
        self.k = k
        self.f = f
        self.noise_level = noise_level
        self.encodings = dict()
        self.prior = None
        self.random_state = random_state
        
    def add_noise(self, series, noise_level):
        return series * (1 + noise_level *   
                         np.random.randn(len(series)))
        
    def fit(self, X, y=None):
        if type(self.categories)=='auto':
            self.categories = np.where(X.dtypes == type(object()))[0]
        
        temp = X.loc[:, self.categories].copy()
        temp['target'] = y
        self.prior = np.mean(y)
        for variable in self.categories:
            avg = (temp.groupby(by=variable)['target']
                       .agg(['mean', 'count']))
            # Compute smoothing 
            smoothing = (1 / (1 + np.exp(-(avg['count'] - self.k) /                 
                         self.f)))
            # The bigger the count the less full_avg is accounted
            self.encodings[variable] = dict(self.prior * (1 -  
                             smoothing) + avg['mean'] * smoothing)
            
        return self
    
    def transform(self, X):
        Xt = X.copy()
        for variable in self.categories:
            Xt[variable].replace(self.encodings[variable], 
                                 inplace=True)
            unknown_value = {value:self.prior for value in 
                             X[variable].unique() 
                             if value not in 
                             self.encodings[variable].keys()}
            if len(unknown_value) > 0:
                Xt[variable].replace(unknown_value, inplace=True)
            Xt[variable] = Xt[variable].astype(float)
            if self.noise_level > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                Xt[variable] = self.add_noise(Xt[variable], 
                                              self.noise_level)
        return Xt
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    

xgb = './xgb_.pkl'

ohe = ['area_name_en',
       'building_name_en',
       'nearest_landmark_en',
      'nearest_metro_en',
      'nearest_mall_en',
      'rooms_en',
      'has_parking']

try:
    df_flat = pd.read_csv('./webui/df_flat.csv',index_col=False)
except:
    print('file not found')
try:
    df_flat = pd.read_csv('./df_flat.csv',index_col=False)
except:
    print('file not found')

df_flat.drop(columns='index', inplace=True)

with st.container():

    area_name_en = tuple(df_flat.area_name_en.unique())
    area_name_en = st.selectbox('Choose area_name_en: ', area_name_en)
    
    building_name_en = tuple(df_flat.building_name_en.unique())
    building_name_en = st.selectbox('Choose building_name_en: ', building_name_en)

    nearest_landmark_en = tuple(df_flat.nearest_landmark_en.unique())
    nearest_landmark_en = st.selectbox('Choose nearest_landmark_en: ', nearest_landmark_en)
 

    nearest_metro_en = tuple(df_flat.nearest_metro_en.unique())
    nearest_metro_en = st.selectbox('Choose nearest_metro_en: ', nearest_metro_en)
 
    nearest_mall_en = tuple(df_flat.nearest_mall_en.unique())
    nearest_mall_en = st.selectbox('Choose nearest_mall_en: ', nearest_mall_en)
 
    has_parking = tuple(df_flat.has_parking.unique())
    has_parking = st.selectbox('has_parking?: ', has_parking)

    rooms_en = tuple(df_flat.rooms_en.unique())
    rooms_en = st.selectbox('rooms_en?: ', rooms_en)


    procedure_area = st.text_input("Procedure area", 100)
    st.write("Procedure area: ", procedure_area)
 

ts_max = df_flat[(df_flat['area_name_en'] == area_name_en) & \
                   (df_flat['building_name_en'] == building_name_en) & \
                   (df_flat['nearest_landmark_en'] == nearest_landmark_en) & \
                   (df_flat['nearest_metro_en'] == nearest_metro_en) & \
                   (df_flat['nearest_mall_en'] == nearest_mall_en) & \
                   (df_flat['has_parking'] == has_parking)]['date'].max()


tmp = df_flat[(df_flat['area_name_en'] == area_name_en) & \
                   (df_flat['building_name_en'] == building_name_en) & \
                   (df_flat['nearest_landmark_en'] == nearest_landmark_en) & \
                   (df_flat['nearest_metro_en'] == nearest_metro_en) & \
                   (df_flat['nearest_mall_en'] == nearest_mall_en) & \
                   (df_flat['has_parking'] == has_parking) & \
                   (df_flat['date'] == ts_max)]

tmp['procedure_area'] = procedure_area
tmp['rooms_en'] = rooms_en

tmp['area_name_freq'] += 1
tmp['building_name_freq'] += 1
tmp['nearest_metro_en_freq'] += 1
tmp['nearest_mall_en_freq'] += 1
tmp['nearest_landmark_en_freq'] += 1


te = TargetEncode(categories=ohe)
te.fit(df_flat, df_flat['actual_worth'])

tmp_encoded = te.transform(tmp)

tmp_encoded.drop(columns = ['year','actual_worth','date'], inplace = True)


print('**************: ', tmp_encoded.head())
X = normalize(np.array(tmp_encoded))
print('>>>>>>>>>>>>>>>> ', X.shape)


xgb = pickle.load(open(xgb, "rb"))

prediction = xgb.predict(X)
print(prediction)

st.write("Prediction >>>> ", prediction)
