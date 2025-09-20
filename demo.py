import cv2
import torch
import numpy as np
import supervision as sv
import sys
sys.path.append("GroundingDINO")
from groundingdino.util.inference import Model
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
# --- NEW IMPORTS ---
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_conversions import convert_color
from deepface import DeepFace
import logging
# Suppress noisy logging from deepface's backends
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('retinaface').setLevel(logging.ERROR)


# =================================================================================
# O'ZBEK TILI UCHUN MATNLAR
# =================================================================================
UZBEK_TEXT = {
    "loading_model": "Asosiy model yuklanmoqda...",
    "model_loaded": "Model muvaffaqiyatli yuklandi.",
    "video_not_found": "Xatolik: 'test.mp4' video fayli topilmadi.",
    "processing_frame": "Kadr {} ishlanmoqda...",
    "detected_total_people": "Jami odamlar: {}",
    "detected_men": "Erkaklar: {}", "detected_women": "Ayollar: {}", "detected_kids": "Bolalar: {}",
    "detected_shoes": "Sotuvdagi poyabzallar: {}",
    "density": "Zichlik: {:.2f} kishi/metr kv",
    "crowding_level": "Tirbandlik darajasi: {}",
    "low": "Past", "medium": "O'rtacha", "high": "Yuqori",
    "heatmap_saved": "Faollik xaritasi 'heatmap.jpg' fayliga saqlandi.",
    "analyzing_faces": "Yuzlar tahlil qilinmoqda...",
    "class_names": { "man": "erkak", "woman": "ayol", "kid": "bola", "shoes": "poyabzal" }
}

# =================================================================================
# SOZLAMALAR
# =================================================================================
GROUNDING_DINO_CONFIG_PATH = "/home/bekhzod/Desktop/SpaceAnalyzer/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/bekhzod/Desktop/SpaceAnalyzer/GroundingDINO/weights/groundingdino_swint_ogc.pth"
CLASSES = ["man", "woman", "kid", "shoes"]
DENSITY_THRESHOLDS = {"high": 0.5, "medium": 0.2}
BOX_TRESHOLD = 0.2
TEXT_TRESHOLD = 0.2
STORE_AREA_SQ_METERS = 50.0
PIXELATION_FACTOR = 30 
ZONE_PADDING = 20 
KID_AGE_THRESHOLD = 16 # Age to classify someone as a "kid"

# =================================================================================
# YORDAMCHI FUNKSIYALAR
# =================================================================================

# --- IMPROVED COLOR CLASSIFICATION ---
COLOR_PALETTE_LAB = {
    "qizil": LabColor(lab_l=53.24, lab_a=80.09, lab_b=67.20),
    "yashil": LabColor(lab_l=46.23, lab_a=-51.70, lab_b=49.90),
    "ko'k": LabColor(lab_l=32.30, lab_a=79.19, lab_b=-107.86),
    "qora": LabColor(lab_l=10.0, lab_a=0.0, lab_b=0.0), # Adjusted for better matching
    "oq": LabColor(lab_l=95.0, lab_a=0.0, lab_b=0.0),   # Adjusted for better matching
    "kulrang": LabColor(lab_l=53.59, lab_a=0.0, lab_b=0.0),
    "sariq": LabColor(lab_l=97.14, lab_a=-21.55, lab_b=94.48),
    "jigarrang": LabColor(lab_l=35.0, lab_a=30.0, lab_b=40.0)
}

def get_dominant_color_lab(image, k=3):
    if image.size == 0: return ""
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = pixels.reshape(-1, 3)
    pixels = np.float32(pixels)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    _, counts = np.unique(labels, return_counts=True)
    dominant_rgb = centers[np.argmax(counts)]
    
    srgb_color = sRGBColor(rgb_r=dominant_rgb[0], rgb_g=dominant_rgb[1], rgb_b=dominant_rgb[2], is_upscaled=True)
    lab_color = convert_color(srgb_color, LabColor)
    
    min_dist = float('inf')
    closest_color_name = ""
    for color_name, lab_palette_color in COLOR_PALETTE_LAB.items():
        dist = delta_e_cie2000(lab_color, lab_palette_color)
        if dist < min_dist:
            min_dist = dist
            closest_color_name = color_name
            
    return closest_color_name

def load_g_dino_model():
    print(UZBEK_TEXT["loading_model"])
    g_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    print(UZBEK_TEXT["model_loaded"])
    return g_dino_model

# --- MODIFIED FOR LIVE HEATMAP ---
def create_heatmap_overlay(background_image, points):
    height, width, _ = background_image.shape
    density_map = np.zeros((height, width), dtype=np.float32)
    
    if not points:
        return np.zeros_like(background_image)

    for x, y in points:
        if 0 <= y < height and 0 <= x < width:
            density_map[y, x] += 1
            
    density_map = gaussian_filter(density_map, sigma=30) # Increased sigma for smoother look
    if np.max(density_map) > 0:
        density_map = (density_map / np.max(density_map) * 255)
        
    heatmap_color = cv2.applyColorMap(density_map.astype(np.uint8), cv2.COLORMAP_JET)
    return heatmap_color

def pixelate_face(face_roi):
    (h, w) = face_roi.shape[:2]
    w_px, h_px = max(1, w // PIXELATION_FACTOR), max(1, h // PIXELATION_FACTOR)
    temp = cv2.resize(face_roi, (w_px, h_px), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

def create_dynamic_zones(shoe_detections, frame_width):
    if len(shoe_detections) == 0:
        return {}
    zones = {}
    frame_center_x = frame_width / 2
    
    left_shoes = shoe_detections[(shoe_detections.xyxy[:, 0] + shoe_detections.xyxy[:, 2]) / 2 < frame_center_x]
    right_shoes = shoe_detections[(shoe_detections.xyxy[:, 0] + shoe_detections.xyxy[:, 2]) / 2 >= frame_center_x]

    if len(left_shoes) > 0:
        x1, y1 = np.min(left_shoes.xyxy[:, [0, 1]], axis=0) - ZONE_PADDING
        x2, y2 = np.max(left_shoes.xyxy[:, [2, 3]], axis=0) + ZONE_PADDING
        zones["chap qator rastasi"] = {"coords": (int(x1), int(y1), int(x2), int(y2)), "color": (255, 165, 0)}

    if len(right_shoes) > 0:
        x1, y1 = np.min(right_shoes.xyxy[:, [0, 1]], axis=0) - ZONE_PADDING
        x2, y2 = np.max(right_shoes.xyxy[:, [2, 3]], axis=0) + ZONE_PADDING
        zones["o'ng qator rastasi"] = {"coords": (int(x1), int(y1), int(x2), int(y2)), "color": (0, 128, 255)}
        
    return zones

# =================================================================================
# ASOSIY FUNKSIYA
# =================================================================================
def main():
    g_dino_model = load_g_dino_model()
    cap = cv2.VideoCapture("test.mp4")
    if not cap.isOpened():
        print(UZBEK_TEXT["video_not_found"])
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('natija_test.mp4', fourcc, fps, (frame_width, frame_height))
    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    heatmap_points = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        print(UZBEK_TEXT["processing_frame"].format(frame_count), end='\r')

        all_detections = g_dino_model.predict_with_classes(
            image=frame, classes=CLASSES, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD
        )

        people_classes_en = {"man", "woman", "kid"}
        people_indices = [i for i, cid in enumerate(all_detections.class_id) if CLASSES[cid] in people_classes_en]
        shoe_indices = [i for i, cid in enumerate(all_detections.class_id) if CLASSES[cid] == "shoes"]
        
        worn_shoe_indices = set()
        if len(people_indices) > 0 and len(shoe_indices) > 0:
            for shoe_idx in shoe_indices:
                shoe_box = all_detections.xyxy[shoe_idx]
                shoe_center_y = (shoe_box[1] + shoe_box[3]) / 2
                for person_idx in people_indices:
                    person_box = all_detections.xyxy[person_idx]
                    if person_box[0] < (shoe_box[0] + shoe_box[2]) / 2 < person_box[2] and \
                       person_box[1] < shoe_center_y < person_box[3] and \
                       shoe_center_y > (person_box[1] + person_box[3]) / 2:
                        worn_shoe_indices.add(shoe_idx)
                        break
        
        for_sale_shoe_indices = [i for i in shoe_indices if i not in worn_shoe_indices]
        for_sale_shoe_detections = all_detections[for_sale_shoe_indices]
        
        final_indices = people_indices + for_sale_shoe_indices
        final_detections = all_detections[final_indices]
        
        # --- NEW: Gender/Age Verification ---
        people_detections = all_detections[people_indices]
        verified_people_classes = {} # Store verified classes {detection_idx: class_name}
        
        if len(people_detections) > 0:
            try:
                # Use deepface on the whole frame once, it's more efficient
                print(f"{UZBEK_TEXT['analyzing_faces']:<30}", end='\r')
                face_analysis = DeepFace.analyze(
                    img_path=frame.copy(), 
                    actions=['age', 'gender'], 
                    enforce_detection=False, 
                    detector_backend='retinaface',
                    silent=True
                )
                
                # Check if face_analysis is a list and contains results
                if isinstance(face_analysis, list) and len(face_analysis) > 0:
                    for person_idx, person_box in zip(people_indices, people_detections.xyxy):
                        person_center_x = (person_box[0] + person_box[2]) / 2
                        person_center_y = (person_box[1] + person_box[3]) / 2
                        
                        for face in face_analysis:
                            face_area = face['region']
                            fx, fy, fw, fh = face_area['x'], face_area['y'], face_area['w'], face_area['h']
                            if fx < person_center_x < fx + fw and fy < person_center_y < fy + fh:
                                if face['age'] < KID_AGE_THRESHOLD:
                                    verified_people_classes[person_idx] = "kid"
                                elif face['dominant_gender'] == 'Man':
                                    verified_people_classes[person_idx] = "man"
                                else:
                                    verified_people_classes[person_idx] = "woman"
                                break # Move to next person once a face is matched

            except Exception as e:
                # This can happen if no faces are found in the frame
                pass
        
        enhanced_labels = []
        for i, (bbox, class_id) in enumerate(zip(final_detections.xyxy, final_detections.class_id)):
            detection_idx = final_indices[i]
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = frame[y1:y2, x1:x2]
            color_name = get_dominant_color_lab(cropped_image)
            
            # --- Use verified class if available ---
            current_class_name_en = verified_people_classes.get(detection_idx, CLASSES[class_id])
            
            current_class_name_uz = UZBEK_TEXT["class_names"].get(current_class_name_en, current_class_name_en)
            new_label = f"{current_class_name_uz} ({color_name})" if color_name else current_class_name_uz
            enhanced_labels.append(new_label)

        # --- Base frame for annotation ---
        annotated_frame = frame.copy()

        # --- NEW: Add Live Heatmap Overlay ---
        heatmap_overlay = create_heatmap_overlay(annotated_frame, heatmap_points)
        annotated_frame = cv2.addWeighted(annotated_frame, 0.6, heatmap_overlay, 0.4, 0)
        
        # Annotate boxes and labels on top of the heatmap overlay
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=final_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=final_detections, labels=enhanced_labels)
        
        # Face pixelation
        try:
            faces = DeepFace.extract_faces(annotated_frame, detector_backend='retinaface', enforce_detection=False)
            for face in faces:
                if face['confidence'] > 0: # Check if a face was actually found
                    fx, fy, fw, fh = face['facial_area'].values()
                    face_roi = annotated_frame[fy:fy+fh, fx:fx+fw]
                    if face_roi.size > 0:
                        pixelated_face = pixelate_face(face_roi)
                        annotated_frame[fy:fy+fh, fx:fx+fw] = pixelated_face
        except Exception:
            pass
            
        dynamic_zones = create_dynamic_zones(for_sale_shoe_detections, frame_width)
        for zone_name, zone_info in dynamic_zones.items():
            coords, color = zone_info["coords"], zone_info["color"]
            x1, y1, x2, y2 = coords
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, zone_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Update counts based on verified classes
        men_count = sum(1 for cid in verified_people_classes.values() if cid == "man")
        women_count = sum(1 for cid in verified_people_classes.values() if cid == "woman")
        kids_count = sum(1 for cid in verified_people_classes.values() if cid == "kid")
        # For unverified people, fall back to GroundingDINO's labels
        unverified_people_indices = [idx for idx in people_indices if idx not in verified_people_classes]
        men_count += sum(1 for i in unverified_people_indices if CLASSES[all_detections.class_id[i]] == "man")
        women_count += sum(1 for i in unverified_people_indices if CLASSES[all_detections.class_id[i]] == "woman")
        kids_count += sum(1 for i in unverified_people_indices if CLASSES[all_detections.class_id[i]] == "kid")
        
        shoes_count = len(for_sale_shoe_detections)
        total_people_count = men_count + women_count + kids_count
        density = total_people_count / STORE_AREA_SQ_METERS if STORE_AREA_SQ_METERS > 0 else 0
        crowding_level = UZBEK_TEXT["low"]
        if density >= DENSITY_THRESHOLDS["high"]: crowding_level = UZBEK_TEXT["high"]
        elif density >= DENSITY_THRESHOLDS["medium"]: crowding_level = UZBEK_TEXT["medium"]

        y_pos = 30
        stats = {
            "detected_total_people": total_people_count, "detected_men": men_count,
            "detected_women": women_count, "detected_kids": kids_count,
            "detected_shoes": shoes_count, "density": density,
            "crowding_level": crowding_level
        }
        
        for key, value in stats.items():
            text = UZBEK_TEXT[key].format(value)
            cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            y_pos += 40
        
        out.write(annotated_frame)
        
        display_frame = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow("Oynani yopish uchun 'q' ni bosing", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Add heatmap points from verified people detections
        current_people_detections = all_detections[[idx for idx in final_indices if CLASSES[all_detections.class_id[idx]] in people_classes_en]]
        for bbox in current_people_detections.xyxy:
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            heatmap_points.append((center_x, center_y))

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nVideo saved as 'natija_test.mp4'")
    
    # Save a final, high-quality heatmap image
    cap = cv2.VideoCapture("test.mp4")
    ret, first_frame = cap.read()
    if ret:
        final_heatmap_overlay = create_heatmap_overlay(first_frame, heatmap_points)
        final_superimposed_img = cv2.addWeighted(first_frame, 0.5, final_heatmap_overlay, 0.5, 0)
        cv2.imwrite("heatmap.jpg", final_superimposed_img)
        print(f"\n{UZBEK_TEXT['heatmap_saved']}")
    cap.release()

if __name__ == "__main__":
    main()
