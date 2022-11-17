from fastai.vision.all import load_learner, torch, PILImage
import streamlit as st
import os
import time
import numpy as np # for image processing 
from PIL import Image


st.set_page_config('flower classification')
st.title('flower classification')
st.markdown ("**Roses, Daisy, Dandelion, Sunflowers, Tulips**")

learn = load_learner('flower.pkl')

def predict(img):
    st.image(img, use_column_width=True)
    with st.spinner('Wait for it...'): time.sleep(3)

    clas, clas_idx, probs = learn.predict(img)
    prob = round(torch.max(probs).item() * 100, 2)
    st.success(f'This is a {clas} with a probability of {prob}%.')


option = st.radio('', ['Take a photo', 'Choose your own image'])

if option == 'Choose your own image':
    uploaded_file = st.file_uploader('Please upload an image', type=['png','jpeg', 'jpg'])
    if uploaded_file is not None:
        img = PILImage.create(uploaded_file)
        predict(img)
        
else:
    file_image = st.camera_input(label = "Take a pic of you to be sketched out")
    if file_image is not None:
        img = PILImage.create(file_image)
        predict(img)

  
from PIL import Image
image = Image.open('sunrise.jpg')

st.image(image, caption='Sunrise by the mountains')
