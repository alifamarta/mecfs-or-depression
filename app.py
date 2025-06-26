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
st.title('Prediksi Diagnosa ME/CFS dan Depression Dengan Metode k-NN ')
k = st.slider('Pilih value dari k untuk k-NN', 1, 15, 5)

# train
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# predict input from user
st.subheader('Prediksi Diagnosa')
input_data = {}


for col in df.columns:
    if col == 'diagnosis':
        continue

    if col == 'age':
        input_data[col] = st.slider('Umur', 18, 100, 40)

    elif col == 'gender':
        input_data[col] = st.selectbox('Jenis Kelamin', ['Male', 'Female'])

    elif col == 'sleep_quality_index':
        input_data[col] = st.slider('Sleep Quality Index (0-10)', 0.0, 10.0, 5.0)

    elif col == 'brain_fog_level':
        input_data[col] = st.slider('Level Brain Fog (0-10)', 0.0, 10.0, 5.0)

    elif col == 'physical_pain_score':
        input_data[col] = st.slider('Skor Kesakitan Fisik (0-10)', 0.0, 10.0, 5.0)

    elif col == 'stress_level':
        input_data[col] = st.slider('Level Stress (0-10)', 0.0, 10.0, 5.0)

    elif col == 'depression_phq9_score':
        input_data[col] = st.slider('Skor Depresi PHQ-9 (0-27)', 0, 27, 10)

    elif col == 'fatigue_severity_scale':
        input_data[col] = st.slider('Skor Skala Kelelahan (0-10)', 0.0, 0.10, 5.0)
    
    elif col == 'pem_duration_hours':
        input_data[col] = st.slider('Durasi PEM (jam)', 0.0, 48.0, 24.0)
    
    # elif col == 'hours_of_sleep_per_night':


    # elif col == 'pem_present':


    # elif col == 'work_status':

    
    # elif col == 'social_activity_level':


    # elif col == 'exercise_frequency':

    
    # elif col == 'meditation_or_mindfulness':


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

    st.success(f'Hasil Prediksi Diagnosa: {pred[0]}')