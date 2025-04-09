# Product Analyzer with Customizable Templates

A Streamlit web application that analyzes product images or videos using AI vision models and generates customized product descriptions in various formats.

## Features

- Upload product images or videos for analysis
- Process media using multiple AI providers:
  - Google Gemini models
  - Open-source models via OpenRouter.ai
- Generate detailed product information including:
  - Product name and category
  - Key features and materials
  - Technical specifications
  - Target audience and benefits
  - Marketing content and social media copy
- Select from multiple description templates:
  - E-commerce Listing
  - Marketing Copy
  - Technical Specification
  - Social Media Post

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/LLM-product-description.git
   cd LLM-product-description
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file or set up Streamlit secrets
   - Add your API keys:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     OPENROUTER_API_KEY=your_openrouter_api_key_here
     ```
   - You can get a Gemini API key from [Google AI Studio](https://ai.google.dev/)
   - You can get an OpenRouter API key from [OpenRouter.ai](https://openrouter.ai/)

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Upload a product image or video file

4. Click "Analyze Product"

5. Select your desired template to format the analysis results

## Advanced Features

The application defaults to using Google's Gemini 2.0 Flash model for analysis, which provides good results for most use cases. Advanced users can access additional options:

### Model Selection (Hidden by Default)

Click on "Advanced Options" to access model selection:

#### Google Gemini Models
- **Gemini 2.0 Flash** - Fast, efficient model for most product analyses
- **Gemini 2.5 Pro** - Advanced model with improved detail recognition

#### Open-source Models
- **Meta Llama 4.0 Maverick** - High-quality image understanding with strong contextual analysis
- **Meta Llama 4.0 Scout** - Optimized version of Llama 4.0
- **MoLMo 7B** - Multimodal model that excels at understanding complex products
- **Qwen 2.5 VL 72B** - Powerful model with excellent image understanding
- **Mistral Small 3.1 24B** - Balanced performance for product analysis

**Note:** Video analysis is currently only supported with Google Gemini models.

## Tips for Best Results

- Use clear, well-lit images with the product as the main focus
- Remove distracting backgrounds when possible
- For complex products, try multiple angles
- For videos, ensure the product is clearly visible
- The analyzer works best with physical products

## Project Structure

- `app.py` - Main Streamlit application
- `templates.py` - Template definitions for various output formats
- `requirements.txt` - Project dependencies

## License

This project is licensed under the terms of the MIT license.