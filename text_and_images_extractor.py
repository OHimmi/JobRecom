import fitz  # PyMuPDF
import streamlit as st
import io
from PIL import Image
import zipfile

def extract_text_and_images_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    images = []
    for page_number in range(len(doc)):
        page = doc[page_number]
        # Extract text
        full_text += page.get_text()

        # Extract images
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Convert to a PIL Image object
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
    return full_text, images

st.title('Job Recommendation System')
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file to disk temporarily (required by PyMuPDF)
    temp_pdf_path = "temp_uploaded_file.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text and images
    extracted_text, images = extract_text_and_images_from_pdf(temp_pdf_path)

    # Display extracted text
    st.write("Extracted Text:")
    st.text_area("Text", value=extracted_text, height=300)

    if images:
        # Save images to a zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for i, image in enumerate(images):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                zip_file.writestr(f'image_{i+1}.jpeg', img_byte_arr.getvalue())
        
        # Download button for all images
        st.download_button(
            label="Download All Images",
            data=zip_buffer.getvalue(),
            file_name="extracted_images.zip",
            mime="application/zip"
        )
    else:
        st.write("No images found in the resume.")
