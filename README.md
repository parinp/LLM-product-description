# Product Analyzer with Customizable Templates

A Streamlit web application that uses Google's Gemini API to analyze product images or videos and generate customized product descriptions in various formats.

## Features

- Upload product images or videos for analysis
- Process media using Gemini's vision capabilities
- Generate detailed product information
- Select from multiple description templates:
  - E-commerce Listing
  - Marketing Copy
  - Technical Specification
  - Social Media Post
- Copy formatted descriptions with one click

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
   ```
   cp .env.example .env
   ```
   
4. Edit the `.env` file and add your Gemini API key. You can get a key from [Google AI Studio](https://ai.google.dev/).

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Upload a product image or video

4. Click "Analyze Product"

5. Select your desired template

6. Copy or download the generated description

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
- `.env` - Environment variables configuration

## License

This project is licensed under the terms of the MIT license.