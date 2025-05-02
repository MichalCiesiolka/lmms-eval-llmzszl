from PIL import Image
import os
import easyocr


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


def extract_polish_text(image_path):
    reader = easyocr.Reader(['pl'])
    result = reader.readtext(image_path)
    plain_text = ' '.join([text for _, text, _ in result])
    return plain_text

def doc_to_text(doc):
    base_path = "../mmllmzszl-test/"  # Adjust this path according to your setup
    PROMPT_BASE = "Answer the Polish exam question from the image. Answer with the good answer letter only. Possible answers are A or B or C or D."
    CONTEXT_PROMPT = "Use the OCR text as well as the image to answer the question. OCR text: "
    # Get the image path from the document
    image_path = os.path.join(base_path, doc["file_name"])
    ocr_text = extract_polish_text(image_path)
    PROMPT = PROMPT_BASE + " " + CONTEXT_PROMPT + ocr_text
    #print(PROMPT)
    return PROMPT

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
