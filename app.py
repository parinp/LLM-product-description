import os
import streamlit as st
from google import genai
from dotenv import load_dotenv
from PIL import Image
import io
import cv2
import numpy as np
import tempfile
import time
from templates import get_template_names, apply_template

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary location and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def upload_video_to_gemini(file_path):
    """Upload a video file to Gemini API and wait until it's processed."""
    # Upload the file to Gemini
    video_file = client.files.upload(file=file_path)
    st.info(f"Video uploaded. Processing...")
    
    # Wait for processing to complete
    while video_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)
        
    # Check if processing was successful
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    
    return video_file

def delete_gemini_file(file_name):
    """Delete a file from Gemini servers."""
    try:
        client.files.delete(name=file_name)
        return True
    except Exception as e:
        st.warning(f"Failed to delete file from Gemini servers: {str(e)}")
        return False

def analyze_media(uploaded_file, file_type):
    """Analyze the uploaded media using Gemini Vision API."""
    try:
        # Create the prompt
        prompt = generate_product_prompt(file_type)
        gemini_file_name = None
        
        # Process based on file type
        if file_type == "image":
            # For images, use direct approach
            image = read_image_file(uploaded_file)
            
            # Generate content using Gemini
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
            
            return_image = image
            return_media_path = None
            
        else:  # video
            # For videos, upload to Gemini
            temp_file_path = save_uploaded_file(uploaded_file)
            
            try:
                # Upload video to Gemini
                video_file = upload_video_to_gemini(temp_file_path)
                gemini_file_name = video_file.name
                
                # Generate content using the video file reference
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt, video_file]
                )
                
                # Delete the file from Gemini servers after analysis
                delete_gemini_file(gemini_file_name)
                
                # Extract a frame for fallback display if needed
                cap = cv2.VideoCapture(temp_file_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    return_image = Image.fromarray(frame_rgb)
                else:
                    # Use a placeholder image if frame extraction fails
                    return_image = Image.new('RGB', (300, 200), color='gray')
                
                # Keep the path to display the video
                return_media_path = temp_file_path
                
            except Exception as e:
                # Ensure temp file is deleted on error
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                
                # Try to delete the file from Gemini if it was created
                if gemini_file_name:
                    delete_gemini_file(gemini_file_name)
                    
                raise e
        
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
            return True, return_image, product_data, return_media_path, file_type
        except json.JSONDecodeError as e:
            return False, return_image, f"Error parsing JSON response: {str(e)}. Raw response: {response.text}", return_media_path, file_type
    
    except Exception as e:
        return False, None, f"Error analyzing media: {str(e)}", None, file_type

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
                success, media_image, result, media_path, media_type = analyze_media(uploaded_file, file_type)
            
            if success:
                # Display results in columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Product Image/Video")
                    
                    # Display media based on type
                    if media_type == "video" and media_path is not None:
                        # Display video player for video files
                        st.video(media_path)
                    else:
                        # Display image for image files or if video display fails
                        st.image(media_image, use_column_width=True)
                    
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
                
                # If we have media to display despite error, show it
                if media_image is not None or media_path is not None:
                    if media_type == "video" and media_path is not None:
                        st.video(media_path)
                    elif media_image is not None:
                        st.image(media_image, caption="Uploaded media")
    
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
    
    # Clean up temporary files when the app is closed
    def cleanup():
        for filename in os.listdir(tempfile.gettempdir()):
            if os.path.isfile(os.path.join(tempfile.gettempdir(), filename)):
                try:
                    os.unlink(os.path.join(tempfile.gettempdir(), filename))
                except:
                    pass
    
    import atexit
    atexit.register(cleanup)

if __name__ == "__main__":
    main() 