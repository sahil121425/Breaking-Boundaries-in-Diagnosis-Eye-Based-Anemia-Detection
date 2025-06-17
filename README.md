import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import joblib
from PIL import Image
import io
from image_processor import preprocess_image, extract_conjunctiva, extract_features
from model import AnemiaPredictionModel
from utils import generate_report
# Set page config
st.set_page_config(
page_title="Anemia Detection",
page_icon="👁️",
layout="wide"
)
# Cache the model loading
@st.cache_resource
def load_model():
model = AnemiaPredictionModel()
model.load_or_train()
print("Using ensemble model for anemia detection")
return model
def analyze_multiple_images(eye_image=None, forniceal_image=None, palpebral_image=None, forniceal_palpebral_image=None, gender="Female", age=30):
"""
Analyze multiple eye images for anemia indicators using ensemble approach.
Args:
eye_image: Eye image (jpg/jpeg)
forniceal_image: Forniceal image (png)
palpebral_image: Palpebral image (png)
forniceal_palpebral_image: Forniceal+Palpebral image (png)
gender: Patient gender ("Male", "Female", or "Child")
age: Patient age in years
Returns:
prediction, confidence, images_processed
"""
try:
# Create a dictionary to hold all available images
images_dict = {}
# Process eye image if available
if eye_image is not None:
file_bytes = np.asarray(bytearray(eye_image.getvalue()), dtype=np.uint8)
eye_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
images_dict['eye'] = eye_img
# Process forniceal image if available
if forniceal_image is not None:
file_bytes = np.asarray(bytearray(forniceal_image.getvalue()), dtype=np.uint8)
forniceal_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
images_dict['forniceal'] = forniceal_img
# Process palpebral image if available
if palpebral_image is not None:
file_bytes = np.asarray(bytearray(palpebral_image.getvalue()), dtype=np.uint8)
palpebral_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
images_dict['palpebral'] = palpebral_img
# Process forniceal+palpebral image if available
if forniceal_palpebral_image is not None:
file_bytes = np.asarray(bytearray(forniceal_palpebral_image.getvalue()), dtype=np.uint8)
forniceal_palpebral_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
images_dict['forniceal_palpebral'] = forniceal_palpebral_img
# If no images were provided, return None
if not images_dict:
return None, None, 0
# Load model
model = load_model()
# Make prediction with ensemble method using all available images and demographic info
prediction, confidence, images_processed = model.predict_from_multiple_images(
images_dict, gender=gender, age=age
)
return prediction, confidence, images_processed
except Exception as e:
st.error(f"Error analyzing images: {str(e)}")
return None, None, 0
# Main function
def main():
# Simple header with a minimalistic title
st.title("Anemia Detection System")
# Main content with clearer instructions
st.markdown("""
### Upload All Four Image Types For Analysis
For accurate anemia detection, please upload as many of these images as possible.
At minimum, the Eye image is required.
""")
# Create tabs for each image type for better organization
tab1, tab2, tab3, tab4 = st.tabs(["👁️ Eye Image", "🔍 Forniceal", "🔍 Palpebral", "🔍 Forniceal+Palpebral"])
with tab1:
st.markdown("### Eye Image (Required)")
st.markdown("Upload a clear image of the eye (JPG format)")
eye_image = st.file_uploader("", type=["jpg", "jpeg"], key="eye")
with tab2:
st.markdown("### Forniceal Image")
st.markdown("Upload a forniceal conjunctiva image (PNG format)")
forniceal_image = st.file_uploader("", type=["png"], key="forniceal")
with tab3:
st.markdown("### Palpebral Image")
st.markdown("Upload a palpebral conjunctiva image (PNG format)")
palpebral_image = st.file_uploader("", type=["png"], key="palpebral")
with tab4:
st.markdown("### Forniceal+Palpebral Image")
st.markdown("Upload a combined forniceal and palpebral image (PNG format)")
forniceal_palpebral_image = st.file_uploader("", type=["png"], key="forniceal_palpebral")
# Add patient demographic information (gender and age)
st.markdown("### Patient Information (Optional)")
st.markdown("This information helps improve prediction accuracy")
col1, col2 = st.columns(2)
with col1:
gender = st.radio("Gender", ["Female", "Male", "Child (<14 years)"])
with col2:
age = st.number_input("Age", min_value=1, max_value=120, value=30)
# Analyze button - full width, more prominent
analyze_button = st.button("ANALYZE ALL IMAGES", type="primary", use_container_width=True)
# Handle image analysis
if analyze_button:
# Check that at least the eye image is uploaded
if eye_image is None:
st.error("Please upload at least the eye image (JPG/JPEG).")
else:
# Show images count message
image_count = sum(x is not None for x in [eye_image, forniceal_image, palpebral_image, forniceal_palpebral_image])
with st.spinner(f"Analyzing {image_count} images using ensemble model..."):
# Process all available images with patient demographic info
prediction, confidence, images_processed = analyze_multiple_images(
eye_image=eye_image,
forniceal_image=forniceal_image,
palpebral_image=palpebral_image,
forniceal_palpebral_image=forniceal_palpebral_image,
gender=gender,
age=age
)
if prediction is not None:
# Display result
st.header("Analysis Results")
# Clear result display
result_text = "ANEMIA DETECTED" if prediction == 1 else "NO ANEMIA DETECTED"
result_color = "#FF5555" if prediction == 1 else "#55AA55"
st.markdown(f"""
<div style="background-color: {result_color}; padding: 40px; border-radius: 10px;
text-align: center; margin: 20px 0;">
<h1 style="color: white; margin: 0; font-size: 32px;">{result_text}</h1>
<p style="color: white; font-size: 22px; margin: 15px 0;">
Confidence: {confidence:.1f}%
</p>
<p style="color: white; font-size: 16px;">
Based on analysis of {images_processed} image types<br>
Patient demographics: {gender}, {age} years old
</p>
</div>
""", unsafe_allow_html=True)
# Recommendation
if prediction == 1: # Anemia detected
st.warning("Medical recommendation: Consider getting a complete blood count (CBC) test to
confirm anemia status.")
# Critical warning for females and children
if gender == "Female" or gender == "Child (<14 years)":
st.error("⚠️ IMPORTANT MEDICAL NOTE: For females with HGB below 11.0 g/dL or children with HGB below 11.5 g/dL, this is CLINICAL ANEMIA that requires medical attention. The lower the HGB, the more severe the anemia.")
st.markdown("""
### Important Information About Anemia
- **Symptoms**: Fatigue, weakness, pale skin, shortness of breath
- **Common causes**: Iron deficiency, vitamin deficiency, chronic diseases
- **Treatment**: Depends on the cause; may include iron supplements, vitamins, or other
Medications
""")
else: # No anemia
st.success("No anemia indicators were detected in the provided images.")
# Safety check - even if the model predicted no anemia, we add warnings for females and children
# This ensures that even if the model is wrong, the user gets proper info
if gender == "Female" or gender == "Child (<14 years)":
st.warning("⚠️ NOTE: Regardless of image analysis results, females with HGB below 11 g/dL or children with HGB below 11.5 g/dL have clinical anemia by definition. If you know HGB values, please rely on laboratory results over image analysis.")
st.markdown("""
### Maintaining Healthy Hemoglobin Levels
- Eat iron-rich foods (lean meats, beans, leafy greens)
- Include vitamin C to help with iron absorption
- Stay hydrated and maintain a balanced diet """)
# Age and Gender considerations - more visible
st.markdown("""
### Hemoglobin Level Guidelines
Normal hemoglobin levels vary by age and gender:
- **Adult males**: 13.5-17.5 g/dL
- **Adult females**: 12.0-15.5 g/dL
- **Pregnant women**: 11.0-16.0 g/dL
- **Children (age-dependent)**: 11.0-16.0 g/dL
Values below these ranges indicate anemia:
- **Mild anemia**: 10.0-11.9 g/dL (females) or 11.0-12.9 g/dL (males)
- **Moderate anemia**: 7.0-9.9 g/dL
- **Severe anemia**: Below 7.0 g/dL
⚠️ **IMPORTANT**: For a female with HGB of 8.0 g/dL, this is MODERATE ANEMIA and
requires medical attention.
""")
else:
st.error("Could not analyze the images. Please try again with clearer images.")
# Display initial message when no images are uploaded
if eye_image is None and forniceal_image is None and palpebral_image is None and forniceal_palpebral_image is None:
st.info("👁️ Please upload images to begin analysis.")
st.markdown(""")
### Instructions
1. Navigate through the tabs to upload all available images
2. The Eye image (JPG) is required, other images are optional but recommended
3. Click the "ANALYZE ALL IMAGES" button
4. View the anemia detection result
For best results, provide all four image types for analysis.
""")
# Simple disclaimer at the bottom
st.markdown("---")
st.caption("⚠️ For educational purposes only. Please consult a healthcare professional for medical
advice.")
if __name__ == "__main__":
main()
