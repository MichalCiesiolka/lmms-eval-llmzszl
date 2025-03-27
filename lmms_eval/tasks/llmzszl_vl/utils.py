from PIL import Image
import os

def llmzszl_doc_to_visual(doc, max_size=(512, 512)):
    # Define the base path where your images are stored
    base_path = "../mmllmzszl-test/"  # Adjust this path according to your setup
    
    # Get the image path from the document
    image_path = os.path.join(base_path, doc["file_name"])
    
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        return None
    
    # Load and return the image
    try:
        img = Image.open(image_path).convert("RGB")
        # Resize the image to reduce memory usage
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return [img]  # Models expect a list of images
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    

def llmzszl_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    metadata = {
        "type": doc["type"],
        "category": doc["name"],
        "needs_image_context": doc["needs_image_context"],
        "year": doc["year"],
    }
    return {"metadata": metadata}
