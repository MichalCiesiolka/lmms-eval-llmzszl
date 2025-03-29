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
    

def parse_pred_ans(pred_ans):
    pred_ans = pred_ans.lower().strip().replace(".", "")
    pred_ans = pred_ans[0]
    return pred_ans
    

def llmzszl_process_results(doc, results):
    metadata = {
        "type": doc["type"],
        "category": doc["name"],
        "needs_image_context": doc["needs_image_context"],
        "year": doc["year"],
    }
    pred = results[0]
    pred_ans = parse_pred_ans(pred)
    gt_ans = doc["correct_answer"].lower().strip().replace(".", "")
    score = 1.0 if pred_ans == gt_ans else 0.0
    prediction = {
        "prediction": pred_ans,
        "correct_answer": gt_ans,
        "score": score,
    }
    return {"metadata": metadata, "prediction": prediction}
