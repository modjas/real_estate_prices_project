import pandas as pd
import joblib
import streamlit as st
import numpy as np

st.title('Predict the price of your apartment')

pipeline = joblib.load('lgbm_pipeline.pkl')

house_type = st.selectbox("Choose apartment type: ", ['kt', 'rt', 'ok'])
sqm = st.slider("Choose apartment size: ", min_value=1, max_value=250, value=50, step=1)
year_built = st.slider("Choose year built: ", min_value=1850, max_value=2020, value=2000, step=1)
elevator = st.selectbox("Choose elevator: ", ['on', 'ei'])
condition = st.selectbox("Choose condition: ", ['hyvä', 'tyyd.', 'huono'])
plot_ownership = st.selectbox("Choose plot ownership: ", ['oma', 'vuokra'])
postal_code = st.text_input("Set postal code: ", value='02500')
floors = st.slider("Choose total floor count: ", min_value=1, max_value=50, value=1, step=1)
current_floor = st.slider("Choose apartment floor number: ", min_value=1, max_value=50, value=1, step=1)
floor_ratio = current_floor/floors

apt = {'house_type' : [house_type],
        'sqm': [sqm],
        'year_built': [year_built],
        'elevator': [elevator],
        'condition': [condition],
        'plot_ownership': [plot_ownership],
        'postal_code': [postal_code],
        'floors': [floors],
        'current_floor': [current_floor],
        'floor_ratio': [floor_ratio]}



prediction = pipeline.predict(pd.DataFrame.from_dict(apt))

prediction_total_value = int(prediction * apt['sqm'])
prediction = int(prediction)

st.write(f"Apartment total value is: {prediction_total_value}€")
st.write(f"Apartment sqm value is: {prediction}€")