import os
import streamlit as st
from google import genai
from dotenv import load_dotenv
from PIL import Image
from PIL import ImageOps
import io
import cv2
import numpy as np
import tempfile
import time
from templates import get_template_names, apply_template

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

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

# Initialize session state variables
if 'current_template' not in st.session_state:
    st.session_state.current_template = None
if 'previous_template' not in st.session_state:
    st.session_state.previous_template = None
if 'show_popup' not in st.session_state:
    st.session_state.show_popup = False
if 'formatted_output' not in st.session_state:
    st.session_state.formatted_output = None
if 'product_data' not in st.session_state:
    st.session_state.product_data = None
if 'regenerating' not in st.session_state:
    st.session_state.regenerating = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "gemini-2.0-flash"

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
    
    # Create a placeholder for status messages
    status_placeholder = st.empty()
    status_placeholder.info("Video uploaded. Processing...")
    
    # Wait for processing to complete
    while video_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = client.files.get(name=video_file.name)
        
    # Clear the processing message
    status_placeholder.empty()
    
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
            
            # Generate content using Gemini with selected model
            response = client.models.generate_content(
                model=st.session_state.selected_model,
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
                
                # Generate content using the video file reference with selected model
                response = client.models.generate_content(
                    model=st.session_state.selected_model,
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

def handle_template_change():
    """Handle template change with confirmation popup."""
    # Get the selected template from the radio button
    new_template = st.session_state.selected_template_radio
    
    # If it's the first selection, just accept it
    if st.session_state.current_template is None:
        st.session_state.current_template = new_template
        st.session_state.formatted_output = apply_template(new_template, st.session_state.product_data)
        return
    
    # If selecting the same template, do nothing
    if new_template == st.session_state.current_template:
        return
        
    # Store the previous and set the new template
    st.session_state.previous_template = st.session_state.current_template
    st.session_state.current_template = new_template
    
    # Show the popup
    st.session_state.show_popup = True

def confirm_template_change():
    """Confirm the template change and regenerate content."""
    # Set regenerating flag and generate new content
    st.session_state.regenerating = True
    st.session_state.formatted_output = apply_template(st.session_state.current_template, st.session_state.product_data)
    st.session_state.show_popup = False
    st.session_state.regenerating = False

def cancel_template_change():
    """Cancel the template change and revert to previous selection."""
    # Revert to previous template
    st.session_state.current_template = st.session_state.previous_template
    st.session_state.show_popup = False
    # Update the radio button value
    st.session_state.selected_template_radio = st.session_state.previous_template

def get_available_models():
    """
    Returns a list of available Gemini models for analysis.
    
    Returns:
        list: List of available model names
    """
    return [
        "gemini-2.0-flash",
        "gemini-2.5-pro-exp-03-25"
    ]

def select_model():
    """
    Creates a model selection dropdown in the Streamlit interface.
    
    Returns:
        str: Selected model name
    """
    available_models = get_available_models()
    selected_model = st.selectbox(
        "Select Model",
        available_models,
        index=available_models.index(st.session_state.selected_model),
        help="Choose the Gemini model to use for analysis"
    )
    return selected_model

def main():
    """Main function to run the Streamlit app."""
    st.title("🔍 Product Analyzer")
    st.write("Upload an image or video of a product to generate customized descriptions.")
    
    # Add instructions at the top
    with st.expander("Tips for Best Results", expanded=True):
        st.markdown("""
        ## Tips for Best Results
        
        - Use clear, well-lit images with the product as the main focus
        - Remove distracting backgrounds when possible
        - For complex products, try multiple angles
        - For videos, ensure the product is clearly visible in the middle frames
        - The analyzer works best with physical products rather than digital goods
        """)
    
    # Add model selection
    st.session_state.selected_model = select_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image or video file", 
        type=["jpg", "jpeg", "png", "mp4", "mov"]
    )
    
    if uploaded_file is not None:
        # Determine file type
        file_type = "video" if uploaded_file.type.startswith("video") else "image"
        
        # Show analysis button
        analyze_clicked = st.button("Analyze Product")
        
        if analyze_clicked:
            # Reset template selection when analyzing a new product
            st.session_state.current_template = None
            st.session_state.previous_template = None
            st.session_state.formatted_output = None
            st.session_state.show_popup = False
            
            with st.spinner(f"Analyzing {file_type}..."):
                success, media_image, result, media_path, media_type = analyze_media(uploaded_file, file_type)
            
            # Store analysis results in session state
            st.session_state.analysis_success = success
            st.session_state.media_image = media_image
            st.session_state.result = result
            st.session_state.media_path = media_path
            st.session_state.media_type = media_type
            
            if success:
                # Store product data in session state for template switching
                st.session_state.product_data = result
        
        # Check if we have analysis results to display
        if hasattr(st.session_state, 'analysis_success') and st.session_state.analysis_success:
            # Get values from session state
            success = st.session_state.analysis_success
            media_image = st.session_state.media_image
            result = st.session_state.result
            media_path = st.session_state.media_path
            media_type = st.session_state.media_type
            
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
                    # Apply EXIF transpose to preserve image orientation
                    media_image = ImageOps.exif_transpose(media_image)
                    st.image(media_image, use_container_width=True)
                
                st.caption(f"Detected Product: {result.get('product_category', 'Unknown')}")
            
            with col2:
                st.subheader("Choose Template")
                
                # Get template names
                template_names = get_template_names()
                
                # Initialize selected_template_radio in session state if not present
                if 'selected_template_radio' not in st.session_state:
                    st.session_state.selected_template_radio = template_names[0]
                
                # Update selected_template_radio if current_template is set but radio isn't matching
                if st.session_state.current_template and st.session_state.selected_template_radio != st.session_state.current_template:
                    st.session_state.selected_template_radio = st.session_state.current_template
                
                # Make template selection using session state for the value
                st.radio(
                    "Select template style:", 
                    template_names,
                    key="selected_template_radio",
                    on_change=handle_template_change
                )
                
                # Update current template if it's not set
                if st.session_state.current_template is None:
                    st.session_state.current_template = st.session_state.selected_template_radio
                    st.session_state.formatted_output = apply_template(st.session_state.current_template, result)
                
                # Display popup if needed
                if st.session_state.show_popup:
                    popup_container = st.container()
                    with popup_container:
                        st.markdown("---")
                        st.subheader("Confirm Template Change")
                        st.write(f"Would you like to regenerate content using the '{st.session_state.current_template}' template?")
                        
                        popup_cols = st.columns(2)
                        with popup_cols[0]:
                            if st.button("Regenerate", key="regenerate_btn", on_click=confirm_template_change):
                                pass  # Logic handled in callback
                        with popup_cols[1]:
                            if st.button("Cancel", key="cancel_btn", on_click=cancel_template_change):
                                pass  # Logic handled in callback
                        st.markdown("---")
                
                # Generate output content
                output_container = st.container()
                with output_container:
                    try:
                        # Show spinner while regenerating
                        if st.session_state.regenerating:
                            with st.spinner("Regenerating content..."):
                                time.sleep(0.5)  # Small delay for UI feedback
                        
                        # If formatted output is in session state, use it
                        if st.session_state.formatted_output:
                            formatted_output = st.session_state.formatted_output
                        else:
                            # Generate for the first time if needed
                            formatted_output = apply_template(st.session_state.current_template, result)
                            st.session_state.formatted_output = formatted_output
                            
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
                
            # Display raw data in expander (for debugging) - full width
            st.markdown("---")
            with st.expander("View Raw Analysis Data"):
                st.json(result)
        elif hasattr(st.session_state, 'analysis_success') and not st.session_state.analysis_success:
            # Show error message
            st.error(st.session_state.result)
            
            # If we have media to display despite error, show it
            if hasattr(st.session_state, 'media_image') and st.session_state.media_image is not None:
                if st.session_state.media_type == "video" and st.session_state.media_path is not None:
                    st.video(st.session_state.media_path)
                else:
                    st.image(st.session_state.media_image, caption="Uploaded media")
    
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