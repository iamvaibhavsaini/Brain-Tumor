import streamlit as st
from PIL import Image
from img_classification import teachable_machine_classification
st.title("Image Classification project")
st.header("Brain Tumor MRI Classification ")
st.text("Upload a brain MRI Image for image classification as tumor or no-tumor")

uploaded_file = st.file_uploader("Choose a brain MRI ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'brain_tumor_classification.h5')
    if label == 0:
        st.write("The MRI scan does not have a brain tumor! you are healthy")
    else:
        st.write("The MRI scan is showing a possible Brain tumor. You need to consult a doctor")
        ##brain_tumor_classification.h5