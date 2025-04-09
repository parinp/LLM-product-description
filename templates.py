"""
Template definitions for product analysis outputs.
Each template provides a structured format for displaying product information.
"""

def format_ecommerce_listing(product_data):
    """
    Format product data as an e-commerce listing with title, features, and description.
    
    Args:
        product_data (dict): Dictionary containing product information
        
    Returns:
        str: Formatted e-commerce listing
    """
    product_name = product_data.get('product_name', 'Product')
    features = product_data.get('features', [])
    description = product_data.get('description', '')
    materials = product_data.get('materials', [])
    
    # Format each feature as a proper markdown list item
    formatted_features = '\n'.join([f"- {feature}" for feature in features])
    
    return f"""# {product_name}

## Key Features
{formatted_features}

## Description
{description}

## Materials
{', '.join(materials)}
"""

def format_marketing_copy(product_data):
    """
    Format product data as persuasive marketing copy with emotional appeals.
    
    Args:
        product_data (dict): Dictionary containing product information
        
    Returns:
        str: Formatted marketing copy
    """
    product_name = product_data.get('product_name', 'Product')
    benefits = product_data.get('benefits', [])
    unique_selling_points = product_data.get('unique_selling_points', [])
    target_audience = product_data.get('target_audience', 'consumers')
    emotional_appeal = product_data.get('emotional_appeal', '')
    
    # Format benefits and unique selling points as proper markdown lists
    benefits_text = '\n'.join([f"- {benefit}" for benefit in benefits])
    usp_text = '\n'.join([f"- {usp}" for usp in unique_selling_points])
    
    return f"""# Introducing the {product_name}

{emotional_appeal}

## Why You'll Love It
{benefits_text}

## What Sets It Apart
{usp_text}

Perfect for {target_audience} who demand nothing but the best.
"""

def format_technical_specification(product_data):
    """
    Format product data as detailed technical specifications.
    
    Args:
        product_data (dict): Dictionary containing product information
        
    Returns:
        str: Formatted technical specifications
    """
    product_name = product_data.get('product_name', 'Product')
    specifications = product_data.get('specifications', {})
    materials = product_data.get('materials', [])
    dimensions = product_data.get('dimensions', '')
    features = product_data.get('features', [])
    
    # Format specifications and features as proper markdown lists
    specs_text = '\n'.join([f"- **{key}**: {value}" for key, value in specifications.items()])
    features_text = '\n'.join([f"- {feature}" for feature in features])
    
    return f"""# Technical Specifications: {product_name}

## Dimensions
{dimensions}

## Materials
{', '.join(materials)}

## Specifications
{specs_text}

## Features
{features_text}
"""

def format_social_media_post(product_data):
    """
    Format product data as a catchy social media post with hashtags.
    
    Args:
        product_data (dict): Dictionary containing product information
        
    Returns:
        str: Formatted social media post
    """
    product_name = product_data.get('product_name', 'Product')
    key_benefit = product_data.get('key_benefit', '')
    call_to_action = product_data.get('call_to_action', 'Check it out!')
    hashtags = product_data.get('hashtags', [])
    
    hashtag_text = ' '.join([f"#{tag.replace(' ', '')}" for tag in hashtags])
    
    return f"""✨ Discover our new {product_name}! ✨

{key_benefit}

{call_to_action}

{hashtag_text}
"""

# Dictionary mapping template names to their formatting functions
TEMPLATES = {
    "E-commerce Listing": format_ecommerce_listing,
    "Marketing Copy": format_marketing_copy,
    "Technical Specification": format_technical_specification,
    "Social Media Post": format_social_media_post
}

def get_template_names():
    """Get a list of available template names."""
    return list(TEMPLATES.keys())

def apply_template(template_name, product_data):
    """
    Apply the specified template to the product data.
    
    Args:
        template_name (str): Name of the template to apply
        product_data (dict): Dictionary containing product information
        
    Returns:
        str: Formatted product information
    """
    if template_name not in TEMPLATES:
        raise ValueError(f"Template '{template_name}' not found")
    
    return TEMPLATES[template_name](product_data) 