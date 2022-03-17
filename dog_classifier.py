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

    
if st.sidebar.button("Click Here to Classify"):

    if CT_Image is None:

        st.sidebar.write("Please upload an Image to Classify")

    else:
        with st.spinner('Classifying ...'):
            Category = ["COVID", "Non-COVID"]
            pred = model.predict((my_image))

            if pred[0][0] > 0.5:
                Cat = "Non-COVID"
            else:
                Cat = "COVID"


            time.sleep(2)
            st.sidebar.success('Done!')

            st.sidebar.header("Algorithm Predicts: ")
            if Cat == "COVID":
                st.sidebar.write("The lung is classified as **COVID-19 affected**")
            elif Cat == "Non-COVID":
                st.sidebar.write("The lung is classified as **Healthy**")
            else:
                st.sidebar.write("ciao")
