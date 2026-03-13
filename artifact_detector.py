import cv2
import numpy as np
from PIL import Image

def detect_artifacts(pil_image: Image.Image) -> dict:
    """
    Detect external objects or imaging artifacts in chest X-rays 
    such as implants, pacemakers, or scanning artifacts.
    
    Uses simple computer vision techniques:
    - edge detection
    - contour detection
    - object bounding boxes
    
    Returns a dictionary containing the detected artifacts, bounding boxes, 
    and an annotated image for visualization.
    """
    # Convert PIL Image to numpy array (RGB to BGR for OpenCV)
    # Ensure image is resized to 224x224 as expected by the pipeline
    rgb = np.array(pil_image.resize((224, 224)).convert("RGB"), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Edge Detection (using Canny)
    # Artifacts like pacemakers and implants usually have sharp, strong and metallic edges
    edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    
    # Apply morphological closing to connect fragmented edges of the same object
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 3. Contour Detection
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to ignore tiny noise and huge anatomical structures
    MIN_AREA = 50
    MAX_AREA = 10000
    valid_contours = []
    bounding_boxes = []
    
    annotated_image = bgr.copy()
    
    # 4. Object bounding boxes
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_AREA < area < MAX_AREA:
            valid_contours.append(contour)
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))
            
            # Draw bounding box on the annotated image
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated_image, "Artifact", (x, max(y - 5, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            
    # Convert annotated image back to RGB for Streamlit visualization
    annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return {
        "annotated_image": annotated_rgb,
        "artifact_contours": valid_contours,
        "bounding_boxes": bounding_boxes,
        "artifact_detected": len(bounding_boxes) > 0,
        "artifact_count": len(bounding_boxes)
    }
