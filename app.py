import time
import streamlit as st
import numpy as np
import pandas as pd

from models.inference import Prediction

st.set_page_config(page_title="Indonesian News Title Category Classifier", page_icon="🗞️", layout="centered")


@st.cache(allow_output_mutation=True, show_spinner=False, ttl=3600, max_entries=10)
def build_model():
    with st.spinner("Loading models... this may take awhile! \n Don't stop it!"):
        inference = Prediction()
    return inference

inference = build_model()

st.title('🗞️ Indonesian News Title Category Classifier')

with st.expander('📋 About this app', expanded=True):
    st.markdown("""
    * Indonesian News Title Category Classifier app is an easy-to-use tool that allows you to predict the category of a given news title.
    * You can predict one title at a time or upload .csv file to bulk predict.
    * Made by [Rizky Adi](https://www.linkedin.com/in/rizky-adi-7b008920b/).
    """)
    st.markdown(' ')

with st.expander('🧠 About prediction model', expanded=False):
    st.markdown("""
    ### Indonesian News Title Category Classifier
    * Model are trained using [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) based on [Indonesian News Title Dataset](https://www.kaggle.com/datasets/ibamibrahim/indonesian-news-title) from Ibrahim on Kaggle.
    * Supported categories are **Finance, Food, Health, Hot, Inet, News, Oto, Sport, Travel**
    * Model test accuracy is **~89%**.
    * **[Source Code](https://github.com/adirizq/indonesian-news-title-category-classifier)**
    """)
    st.markdown(' ')


st.markdown(' ')
st.markdown(' ')

st.header('🔍 News Title Category Prediction')

title = st.text_input('News Title', placeholder='Enter your shocking news title')

if title:
    with st.spinner('Loading prediction...'):
        result = inference.predict(title)
    st.markdown(f'Category for this news is **[{result}]**')


st.markdown(' ')
st.markdown(' ')

st.header('🗃️ Bulk News Title Category Prediction')
st.markdown('Only upload .csv file that contains list of news titles separated by comma.')

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, header=None)
    results = []

    with st.spinner('Loading prediction...'):
        for title in df[0]:
            result = inference.predict(title)
            results.append({'Title': title, 'Category': result})

        df_results = pd.DataFrame(results)

    st.markdown('#### Prediction Result')
    st.download_button(
        "Download Result",
        df_results.to_csv(index=False).encode('utf-8'),
        "News Title Category Prediction Result.csv",
        "text/csv",
        key='download-csv'
    )
    st.dataframe(df_results, 1000)
 
