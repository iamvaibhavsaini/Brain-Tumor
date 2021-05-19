import streamlit as st
from PIL import Image
from img_classification import teachable_machine_classification
st.title("Brain Tumor Detection using ConvNets")
st.header("Based on a public dataset of MRI scans")
image = Image.open('Logo1.jpg')
st.image(image, caption='Brain Tumor Detection using ConvNets',use_column_width=True)
st.text("Upload an MRI scan of a person who may be unsure of seeking further medical counsel, and our algorithm will help you in deciding if to seek further treatment")

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'brain_tumor_classification.h5')
    if label == 0:
        st.write("The MRI scan shows no abnormal growths, your brain scan is quite Normal.")
    else:
        st.write("There is an abnormal growth detected in your MRI brain scan. Check with your local neurologist.")
        ##brain_tumor_classification.h5
