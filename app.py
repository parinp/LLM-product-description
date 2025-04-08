import os
import streamlit as st
from google import genai
from dotenv import load_dotenv
from PIL import Image
import io
import cv2
import numpy as np
import tempfile
from templates import get_template_names, apply_template

# Load environment variables
load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please add it to your .env file.")
    st.stop()

# Configure Gemini API
client = genai.Client(api_key=GEMINI_API_KEY)

# Set page config
st.set_page_config(
    page_title="Product Analyzer",
    page_icon="🔍",
    layout="wide",
)

def generate_product_prompt(file_type="image"):
    """Generate a prompt for product analysis based on file type."""
    return f"""
    You are a product analysis expert. Please analyze this {file_type} and provide the following information in JSON format:
    
    {{
        "product_name": "Descriptive product title",
        "product_category": "Category and subcategory of the product",
        "features": ["List of 3-5 key visible features"],
        "materials": ["List of visible materials used"],
        "description": "A paragraph describing the product in detail",
        "specifications": {{"spec1": "value1", "spec2": "value2"}},
        "dimensions": "Approximate dimensions if visible",
        "benefits": ["List of 3-5 potential user benefits"],
        "unique_selling_points": ["List of 2-3 unique selling points"],
        "target_audience": "Description of likely target audience",
        "emotional_appeal": "A sentence appealing to emotions for marketing",
        "key_benefit": "The single most important benefit for social media",
        "call_to_action": "A call to action for social media",
        "hashtags": ["List of 3-5 relevant hashtags for social media"]
    }}
    
    Be accurate and specific in your analysis. If you cannot determine certain information, provide your best educated guess.
    The response MUST be valid JSON that can be parsed.
    """

def read_image_file(uploaded_file):
    """Read an image file and return a PIL Image object."""
    # Read bytes from uploaded file
    bytes_data = uploaded_file.getvalue()
    
    # Convert bytes to image
    img = Image.open(io.BytesIO(bytes_data))
    return img

def read_video_frame(uploaded_file):
    """Extract a frame from the video file."""
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Open video file
    cap = cv2.VideoCapture(tmp_file_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        os.unlink(tmp_file_path)
        raise ValueError("Failed to open video file")
    
    # Read middle frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    ret, frame = cap.read()
    
    # Release video capture and delete temp file
    cap.release()
    os.unlink(tmp_file_path)
    
    if not ret:
        raise ValueError("Failed to extract frame from video")
    
    # Convert from BGR to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    
    return img

def analyze_media(uploaded_file, file_type):
    """Analyze the uploaded media using Gemini Vision API."""
    try:
        # Process based on file type
        if file_type == "image":
            image = read_image_file(uploaded_file)
        else:  # video
            image = read_video_frame(uploaded_file)
        
        # Create the prompt
        prompt = generate_product_prompt(file_type)
        
        # Generate content using Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image]
        )
        
        # Parse JSON from the response
        import json
        try:
            # Extract JSON string from response
            response_text = response.text
            # Check if the response is wrapped in ```json and ``` markers
            if "```json" in response_text and "```" in response_text.split("```json", 1)[1]:
                json_str = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
            else:
                json_str = response_text
            
            # Parse JSON
            product_data = json.loads(json_str)
            return True, image, product_data
        except json.JSONDecodeError as e:
            return False, image, f"Error parsing JSON response: {str(e)}. Raw response: {response.text}"
    
    except Exception as e:
        return False, None, f"Error analyzing media: {str(e)}"

def main():
    """Main function to run the Streamlit app."""
    st.title("🔍 Product Analyzer")
    st.write("Upload an image or video of a product to generate customized descriptions.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file", 
        type=["jpg", "jpeg", "png", "mp4", "mov"]
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_type = "video" if uploaded_file.type.startswith("video") else "image"
        
        # Show analysis button
        if st.button("Analyze Product"):
            with st.spinner(f"Analyzing {file_type}..."):
                success, media, result = analyze_media(uploaded_file, file_type)
            
            if success:
                # Display results in columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Product Image")
                    st.image(media, use_column_width=True)
                    st.caption(f"Detected Product: {result.get('product_category', 'Unknown')}")
                
                with col2:
                    st.subheader("Choose Template")
                    template_names = get_template_names()
                    selected_template = st.radio("Select template style:", template_names)
                    
                    try:
                        formatted_output = apply_template(selected_template, result)
                        st.markdown(formatted_output)
                        
                        # Add copy button
                        st.download_button(
                            label="Copy to clipboard",
                            data=formatted_output,
                            file_name=f"{result.get('product_name', 'product').replace(' ', '_')}.md",
                            mime="text/markdown"
                        )
                    except Exception as e:
                        st.error(f"Error applying template: {str(e)}")
                
                # Display raw data in expander (for debugging)
                with st.expander("View Raw Analysis Data"):
                    st.json(result)
            else:
                st.error(result)
    
    # Add instructions at the bottom
    with st.expander("Tips for Best Results"):
        st.markdown("""
        ## Tips for Getting the Best Results
        
        - Use clear, well-lit images with the product as the main focus
        - Remove distracting backgrounds when possible
        - For complex products, try multiple angles
        - For videos, ensure the product is clearly visible in the middle frames
        - The analyzer works best with physical products rather than digital goods
        """)

if __name__ == "__main__":
    main() 