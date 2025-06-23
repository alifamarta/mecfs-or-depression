import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

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

# streamlit 
st.title('Prediksi Diagnosis ME/CFS dan Depression Dengan Metode k-NN ')
k = st.slider('Pilih value dari K untuk k-NN', 1, 15, 5)

# train
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# predict input from user
st.subheader('Prediksi Diagnosis')
input_data = {}

for col in df.columns:
    if col == 'diagnosis':
        continue

    if x[col].dtype == 'int64' or x[col].dtype == 'float64':
        input_data[col] = st.number_input(col, value=float(x[col].median()))
    
    elif col == 'gender':
        input_data[col] = st.selectbox(col, options=['Male', 'Female'])
        
    else:
        input_data[col] = st.text_input(col, value=str(x[col].mode()[0]))

if st.button('Prediksi'):
    input_df = pd.DataFrame([input_data])

    # encode & scaling input from user
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = le.fit_transform(input_df[col])
    
    for col in input_df.select_dtypes(include=['float64', 'int64']).columns:
        input_df[col] = input_df[col].fillna(x[col].median())

    input_df = scaler.transform(input_df)
    pred = knn.predict(input_df)

    st.success(f'Hasil Prediksi Diagnosis: {pred[0]}')