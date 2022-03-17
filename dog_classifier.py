from fastai.vision.all import load_learner, torch, PILImage
import streamlit as st
import os
import time

st.set_page_config('PETS', '🐶&🐱')
st.title('PETS 🐶&🐱')

learn = load_learner('flower.pkl')

def predict(img):
    st.image(img, use_column_width=True)
    with st.spinner('Wait for it...'): time.sleep(3)

    clas, clas_idx, probs = learn.predict(img)
    prob = round(torch.max(probs).item() * 100, 2)
    st.success(f'This is a {clas} with a probability of {prob}%.')


    uploaded_file = st.file_uploader('Please upload an image', type='jpg')
    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        predict(img)
