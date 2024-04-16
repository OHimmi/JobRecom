import streamlit as st
import pdfplumber
from PIL import Image
import io

def extract_text_and_images_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        full_text = ''
        images = []
        for page in pdf.pages:
            full_text += page.extract_text() if page.extract_text() else ''
            for img in page.images:
                # Extract images
                image_obj = page.to_image()
                bbox = (img['x0'], img['top'], img['x1'], img['bottom'])
                cropped_image = image_obj.crop(bbox)
                images.append(cropped_image.image.convert('RGB'))
        return full_text, images

st.title('Job Recommendation System')
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf"])

if uploaded_file is not None:
    text, images = extract_text_and_images_from_pdf(uploaded_file)
    st.write("Extracted Text:", text)
    
    if images:
        for i, image in enumerate(images):
            st.image(image, caption=f"Image {i+1}")
            # Save image to a bytes buffer
            buf = io.BytesIO()
            image.save(buf, format='JPEG')
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name=f"image_{i+1}.jpg",
                mime="image/jpeg"
            )
    else:
        st.write("No images found in the resume.")
    # Add your recommendation logic here based on the extracted text
