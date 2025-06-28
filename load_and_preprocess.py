import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

@st.cache_data
def load_and_preprocess():
    df = pd.read_csv('dataset/mecfs_vs_depression_dataset.csv')

    # data preprocessing
    df = df.dropna(subset=['diagnosis'])
    x = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # fill missing value
    for col in x.select_dtypes(include=['object']).columns:
        x[col] = x[col].fillna('Unknown')
        le = LabelEncoder()
        x[col] = le.fit_transform(x[col])

    for col in x.select_dtypes(include=['float64', 'int64']).columns:
        x[col] = x[col].fillna(x[col].median())

    # split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    return df, x, y, x_train, x_test, y_train, y_test, scaler