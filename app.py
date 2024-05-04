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

try:
    df_flat = pd.read_csv('./webui/df_flat.csv')
except:
    print('file not found')
try:
    df_flat = pd.read_csv('./df_flat.csv')
except:
    print('file not found')


add_page_title()


show_pages(
    [
        Page("./app.py", "Home", "🏠"),

        Section(name = "Analytics", icon="📊"),
        Page("./webui/pages/Region_Analytics.py", "Region Analytics", "🌍"),
        Page("./webui/pages/Unit_Analytics.py", "Unit Analytics", "🏛"),  
        
        Section(name = "Predictive Tools", icon="📈"),
        Page("./webui/pages/Prediction_Tools.py", "Predictive Tools", "📈"),

        Section(name = "About us", icon="🚀"),
        Page("./webui/pages/About.py", "About us", "🚀")
        
    ]
)
with st.container():

   st.markdown("""
    <style>
    .big-font {
    font-size:50px !important;
    }
    </style>
    """, unsafe_allow_html=True)

   st.markdown('<p class="big-font">Welcome to AI Invest Property Dubai!</p>', unsafe_allow_html=True)

   st.image('./webui/imgs/dubai_palm.jpg')


