from fastai.vision.all import load_learner, torch, PILImage
import streamlit as st
import os
import time

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
    
st.radio('', ['Choose your own image'])

uploaded_file = st.file_uploader('Please upload an image', type=['png','jpeg', 'jpg'])

if uploaded_file is not None:
 img = PILImage.create(uploaded_file)
 predict(img)

    

 import cv2
import streamlit as st

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
