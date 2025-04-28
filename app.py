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
import json
import base64
from openai import OpenAI
from templates import get_template_names, apply_template

# Load environment variables
load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]


if not GEMINI_API_KEY or not OPENROUTER_API_KEY:
    st.error("API keys not found. Please add them to your .env file.")
    st.stop()

# Initialize Gemini API
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize OpenRouter client
openrouter_client = None
if OPENROUTER_API_KEY:
    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

# Set page config
st.set_page_config(
    page_title="Product Analyzer",
    page_icon="static/analyze.png",
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
    """Generate a prompt for product analysis based on file type and return a string in json format"""
    return f"""
    You are a product analysis expert. Please analyze this {file_type} and provide the following information in JSON format:
    
    {{
        "product_name": "Descriptive product title",
        "product_category": "Category and subcategory of the product",
        "features": ["List of 3-5 key visible features"],
        "materials": ["List of visible materials used"],
        "description": "A paragraph describing the product in detail. Ensure the description is concise and to the point within a few sentences that would take up 2-3 lines of text.",
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
    bytes_data = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(bytes_data))
    return img

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to a temporary location and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def upload_video_to_gemini(file_path):
    """Upload a video file to Gemini API and wait until it's processed."""
    
    video_file = gemini_client.files.upload(file=file_path)
    
    # Status placeholder
    status_placeholder = st.empty()
    status_placeholder.info("Video uploaded. Processing...")
    
    # Wait
    while video_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = gemini_client.files.get(name=video_file.name)
        
    # Clear messages
    status_placeholder.empty()
    
    # Check if successful
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    
    return video_file

def delete_gemini_file(file_name):
    """Delete a file from Gemini servers."""
    try:
        gemini_client.files.delete(name=file_name)
        return True
    except Exception as e:
        st.warning(f"Failed to delete file from Gemini servers: {str(e)}")
        return False

def analyze_with_openrouter(image, model_name, prompt):
    """
    Analyze the image using OpenRouter API.
    
    Args:
        image (PIL.Image): Image to analyze
        model_name (str): OpenRouter model ID to use
        prompt (str): Prompt for the model
        
    Returns:
        str: Response from the model
    """
    if not openrouter_client:
        raise ValueError("Open-source models API key not configured")
    
    # Convert image to RGB mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    try:
        response = openrouter_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2048
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Open-source models API error: {str(e)}")

def analyze_media(uploaded_file, file_type):
    """Analyze the uploaded media using either Gemini Vision API or OpenRouter API."""
    try:
        # Create the prompt
        prompt = generate_product_prompt(file_type)
        gemini_file_name = None
        selected_model = st.session_state.selected_model
        
        # Check if open-source model is selected
        use_openrouter = is_openrouter_model(selected_model)
        
        # Process based on file type
        if file_type == "image":
            # For images, process based on selected API
            image = read_image_file(uploaded_file)
            
            if use_openrouter:
                if not OPENROUTER_API_KEY:
                    return False, image, "Open-source models API key not configured. Please add it to your .env file.", None, file_type
                
                # Use OpenRouter for analysis
                try:
                    response_text = analyze_with_openrouter(image, selected_model, prompt)
                except Exception as e:
                    return False, image, f"Error with OpenRouter API: {str(e)}", None, file_type
            else:
                # Generate content using Gemini
                response = gemini_client.models.generate_content(
                    model=selected_model,
                    contents=[prompt, image]
                )
                response_text = response.text
            
            return_image = image
            return_media_path = None
            
        else:  # video
            # Videos are only supported in Gemini currently
            if use_openrouter:
                return False, None, "Video analysis is currently only supported with Google Gemini models. Please select a Gemini model.", None, file_type
            
            # For videos, upload to Gemini
            temp_file_path = save_uploaded_file(uploaded_file)
            
            try:
                # Upload video to Gemini
                video_file = upload_video_to_gemini(temp_file_path)
                gemini_file_name = video_file.name
                
                # Generate content using the video file reference
                response = gemini_client.models.generate_content(
                    model=selected_model,
                    contents=[prompt, video_file]
                )
                response_text = response.text
                
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
        # This step is necessary because free models are unable to generated structured outputs
        try:
            # Extract JSON string from response
            # Check if the response is wrapped in ```json and ``` markers
            if "```json" in response_text and "```" in response_text.split("```json", 1)[1]:
                json_str = response_text.split("```json", 1)[1].split("```", 1)[0].strip()
            else:
                json_str = response_text
            
            # Parse JSON
            product_data = json.loads(json_str)
            return True, return_image, product_data, return_media_path, file_type
        except json.JSONDecodeError as e:
            return False, return_image, f"Error parsing JSON response: {str(e)}. Raw response: {response_text}", return_media_path, file_type
    
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
    Returns a list of available models for analysis.
    
    Returns:
        list: List of available model names
    """
    gemini_models = [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro-exp-03-25"
    ]
    
    openrouter_models = [
        "meta-llama/llama-4-maverick:free",
        "meta-llama/llama-4-scout:free",
        "allenai/molmo-7b-d:free",
        "qwen/qwen2.5-vl-72b-instruct:free",
        "mistralai/mistral-small-3.1-24b-instruct:free"
    ]

    return gemini_models + openrouter_models

def is_openrouter_model(model_name):
    """
    Checks if the given model is available through OpenRouter.
    
    Args:
        model_name (str): Name of the model to check
        
    Returns:
        bool: True if the model is an OpenRouter model, False otherwise
    """
    return model_name not in ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro-exp-03-25"]

def select_model():
    """
    Creates a model selection interface in the Streamlit app,
    grouping models by their provider.
    
    Returns:
        str: Selected model name
    """
    # Get all available models
    all_models = get_available_models()
    
    # Group models by provider
    gemini_models = [m for m in all_models if not is_openrouter_model(m)]
    openrouter_models = [m for m in all_models if is_openrouter_model(m)]
    
    # Provider selection
    provider = st.radio(
        "Select Provider",
        ["Google Gemini", "Open-source models"],
        index=0 if not is_openrouter_model(st.session_state.selected_model) else 1,
        help="Choose the AI provider"
    )
    
    if provider == "Google Gemini":
        available_models = gemini_models
        # Default to first Gemini model if current selection is not a Gemini model
        default_index = 0
        if st.session_state.selected_model in gemini_models:
            default_index = gemini_models.index(st.session_state.selected_model)
            
        selected_model = st.selectbox(
            "Select Gemini Model",
            available_models,
            index=default_index,
            help="Choose the Gemini model to use for analysis"
        )
        
        # Show Gemini model description
        if "gemini-2.0-flash" in selected_model:
            st.info("Gemini 2.0 Flash is a fast and efficient model for image analysis with good accuracy for most use cases.")
        elif "gemini-2.5-flash" in selected_model:
            st.info("Gemini 2.5 Flash is a balanced model that combines speed with enhanced detail recognition, ideal for quick yet precise product analysis.")
        elif "gemini-2.5" in selected_model:
            st.info("Gemini 2.5 Pro is an advanced model with improved detail recognition and better understanding of complex products.")
        
    else:
        if not OPENROUTER_API_KEY:
            st.error("Open-source models API key not configured. Please add it to your .env file.")
            if openrouter_models:
                selected_model = st.selectbox(
                    "Select Open-source Model (disabled)",
                    openrouter_models,
                    index=0,
                    disabled=True
                )
                selected_model = st.session_state.selected_model  # Keep current selection
            else:
                selected_model = st.session_state.selected_model  # Keep current selection
        else:
            available_models = openrouter_models
            # Default to first OpenRouter model if current selection is not an OpenRouter model
            default_index = 0
            if st.session_state.selected_model in openrouter_models:
                default_index = openrouter_models.index(st.session_state.selected_model)
                
            model_display_names = {
                "meta-llama/llama-4-maverick:free": "Meta Llama 4.0 Maverick",
                "meta-llama/llama-4-scout:free": "Meta Llama 4.0 Scout",
                "allenai/molmo-7b-d:free": "MoLMo 7B",
                "qwen/qwen2.5-vl-72b-instruct:free": "Qwen 2.5 VL 72B",
                "mistralai/mistral-small-3.1-24b-instruct:free": "Mistral Small 3.1 24B"
            }
            
            # Create a list of display names in the same order as available_models
            display_options = [model_display_names.get(model, model) for model in available_models]
            
            # Create a mapping from display name back to model ID
            display_to_model = {display: model for display, model in zip(display_options, available_models)}
            
            selected_display = st.selectbox(
                "Select Open-source Model",
                display_options,
                index=default_index,
                help="Choose the open-source model to use for analysis"
            )
            
            # Convert the display name back to the model ID
            selected_model = display_to_model[selected_display]
            
            # Always show model description for selected OpenRouter model
            if "llama-4" in selected_model:
                st.info("Meta Llama 4 offers high-quality image understanding with strong contextual analysis.")
            elif "molmo" in selected_model:
                st.info("MoLMo is a multimodal model that excels at understanding complex products with good visual detail recognition.")
            elif "qwen" in selected_model:
                st.info("Qwen 2.5 VL is a powerful multimodal model with excellent image understanding and detailed descriptions.")
            elif "mistral" in selected_model:
                st.info("Mistral Small provides balanced performance for product analysis with efficient processing.")
    
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
    
    # Add model selection as an advanced option (hidden by default)
    with st.expander("Advanced Options", expanded=False):
        st.markdown("### Model Selection (Advanced)")
        st.write("Select which AI model to use for image analysis. Different models may provide varying levels of detail and accuracy.")
        st.session_state.selected_model = select_model()
    
    # If no model is explicitly selected, use the default
    if 'selected_model' not in st.session_state or st.session_state.selected_model is None:
        st.session_state.selected_model = "gemini-2.0-flash"
    
    # Show a small indicator of the currently selected model
    selected_model_name = st.session_state.selected_model
    # Get a user-friendly name
    if is_openrouter_model(selected_model_name):
        # For OpenRouter models, use their friendly names
        model_display_names = {
            "meta-llama/llama-4-maverick:free": "Meta Llama 4.0 Maverick",
            "meta-llama/llama-4-scout:free": "Meta Llama 4.0 Scout",
            "allenai/molmo-7b-d:free": "MoLMo 7B",
            "qwen/qwen2.5-vl-72b-instruct:free": "Qwen 2.5 VL 72B", 
            "mistralai/mistral-small-3.1-24b-instruct:free": "Mistral Small 3.1 24B"
        }
        display_name = model_display_names.get(selected_model_name, selected_model_name)
        st.caption(f"Using model: {display_name} (open-source)")
    else:
        # For Gemini models, format the name nicely
        if "gemini-2.0-flash" in selected_model_name:
            display_name = "Gemini 2.0 Flash"
        elif "gemini-2.5-flash" in selected_model_name:
            display_name = "Gemini 2.5 Flash"
        elif "gemini-2.5" in selected_model_name:
            display_name = "Gemini 2.5 Pro"
        else:
            display_name = selected_model_name
        st.caption(f"Using model: {display_name}")

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