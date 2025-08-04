import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from load_and_preprocess import load_and_preprocess

df, x, y, x_train, x_test, y_train, y_test, scaler = load_and_preprocess()

# train
knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan')
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

# streamlit 
st.title('Prediksi Diagnosis ME/CFS dan Depression Dengan Metode k-NN ')

st.markdown('<hr>', unsafe_allow_html=True)

# about
st.markdown('''
## Tentang Aplikasi
Aplikasi ini adalah sebuah aplikasi web yang dirancang untuk membedakan antara membedakan antara ME/CFS (Myalgic Encephalomyelitis/Chronic Fatigue Syndrome) dan depresi berdasarkan data gejala yang dimasukkan pengguna.

Dengan bantuan dari sebuah dataset dari kaggle, aplikasi ini akan memprediksi diagnosis ME/CFS atau depresi berdasarkan data gejala yang dimasukkan oleh pengguna.

Sumber dataset: [ME/CFS vs Depression Classification Dataset](https://www.kaggle.com/datasets/storytellerman/mecfs-vs-depression-classification-dataset)
 ''')

# medical explanation

st.markdown('<hr>', unsafe_allow_html=True)

st.markdown(''' 
## Penjelasan Medis
            
- **ME/CFS**: Myalgic Encephalomyelitis/Chronic Fatigue Syndrome (Kelelahan Kronis)
- **PEM**: Post-Exertional Malaise (Memburuknya gejala setelah aktivitas fisik/mental)
- **Brain Fog**: Kesulitan untuk berpikit, berkonsentrasi, atau mengingat 
- **PHQ-9**: Kuesioner standar untuk menilai tingkat depresi
''')

st.markdown('<hr>', unsafe_allow_html=True)

# predict input from user
st.subheader('Isi Form Ini Untuk Melakukan Prediksi Diagnosis')
input_data = {}

# Mapping
gender_map = {'Laki-Laki': 'Male', 'Perempuan': 'Female'}

pem_present_map = {'Ya': '1', 'Tidak': '1'}

work_status_map = {
    'Bekerja': 'Working',
    'Tidak Bekerja': 'Not Working',
    'Bekerja Paruh Waktu': 'Partially Working'
}

social_activity_map = {
    'Sangat Tinggi': 'Very High',
    'Tinggi': 'Tinggi',
    'Sedang': 'Medium', 
    'Rendah': 'Low',
    'Sangat Rendah': 'Very Low'
}

exercise_map = {
    'Setiap Hari': 'Daily',
    'Sering': 'Often',
    'Kadang-Kadang': 'Sometimes',
    'Jarang': 'Rarely',
    'Tidak Pernah': 'Never'
}

meditation_map = {'Ya': 'Yes', 'Tidak': 'No'}

for col in df.columns:
    if col == 'diagnosis':
        continue

    if col == 'age':
        input_data[col] = st.slider('Umur', 18, 100, 40)

    elif col == 'gender':
        gender = st.selectbox('Jenis Kelamin', list(gender_map.keys()))
        input_data[col] = gender_map[gender]

    elif col == 'sleep_quality_index':
        input_data[col] = st.slider('Sleep Quality Index (0-10)', 0.0, 10.0, 5.0)

    elif col == 'brain_fog_level':
        input_data[col] = st.slider('Level Brain Fog (0-10)', 0.0, 10.0, 5.0)

    elif col == 'physical_pain_score':
        input_data[col] = st.slider('Skor Nyeri Fisik (0-10)', 0.0, 10.0, 5.0)

    elif col == 'stress_level':
        input_data[col] = st.slider('Level Stress (0-10)', 0.0, 10.0, 5.0)

    elif col == 'depression_phq9_score':
        input_data[col] = st.slider('Skor Depresi PHQ-9 (0-27)', 0, 27, 10)

    elif col == 'fatigue_severity_scale_score':
        input_data[col] = st.slider('Skor Kelelahan (0-10)', 0.0, 10.0, 5.0)
    
    elif col == 'pem_duration_hours':
        input_data[col] = st.slider('Durasi PEM (jam)', 0.0, 48.0, 24.0)
    
    elif col == 'hours_of_sleep_per_night':
        input_data[col] = st.slider('Jam Tidur Per Malam', 0.0, 12.0, 7.0)

    elif col == 'pem_present':
        input_data[col] = st.selectbox('PEM Present', list(pem_present_map.keys()))

    elif col == 'work_status':
        work_status = st.selectbox('Status Kerja', list(work_status_map.keys()))
        input_data[col] = work_status_map[work_status]
    
    elif col == 'social_activity_level':
        social_activity = st.selectbox('Level Aktivitas Sosial', list(social_activity_map.keys()))
        input_data[col] = social_activity_map[social_activity]

    elif col == 'exercise_frequency':
        exercise = st.selectbox('Frekuensi Olahraga', list(exercise_map.keys()))
        input_data[col] = exercise_map[exercise]
    
    elif col == 'meditation_or_mindfulness':
        meditation = st.selectbox('Meditasi/Mindfulness', list(meditation_map.keys()))
        input_data[col] = meditation_map[meditation]

if st.button('Prediksi'):
    input_df = pd.DataFrame([input_data])

    # encode & scaling input from user
    for col in input_df.select_dtypes(include=['object']).columns:
        input_df[col] = LabelEncoder().fit_transform(input_df[col])
    
    for col in input_df.select_dtypes(include=['float64', 'int64']).columns:
        input_df[col] = input_df[col].fillna(x[col].median())

    input_df = scaler.transform(input_df)
    pred = knn.predict(input_df)

    diagnosis_map = {
        'Depression': 'Depresi',
        'ME/CFS': 'ME/CFS',
        'Both': 'Keduanya (Depresi dan ME/CFS)'
    }

    diagnosis = diagnosis_map.get(pred[0], pred[0])
    st.success(f'Hasil Prediksi Diagnosis: {diagnosis}')

# evaluasi model
with st.expander('Evaluasi Model'):
    accuracy = knn.score(x_test, y_test)
    st.write(f'**Akurasi pada data yang telah diuji:** {accuracy:.2%}')
    cm = confusion_matrix(y_test, y_pred)
    st.write('**Confusion Matrix:**')
    st.write(pd.DataFrame(cm, index=knn.classes_, columns=knn.classes_))

    st.write('**Classification Report:**')
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())
