import cv2
import torch
import numpy as np
import supervision as sv
from retinaface import RetinaFace
import sys
sys.path.append("GroundingDINO")
from groundingdino.util.inference import Model
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

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
    "colors": {
        "qizil": ([0, 120, 70], [10, 255, 255]), "yashil": ([36, 100, 100], [86, 255, 255]),
        "ko'k": ([94, 80, 2], [126, 255, 255]), "qora": ([0, 0, 0], [180, 255, 30]),
        "oq": ([0, 0, 200], [180, 20, 255]), "kulrang": ([0, 0, 40], [180, 20, 200])
    },
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
ZONE_PADDING = 20 # Dinamik zonalarga qo'shiladigan bo'sh joy

# =================================================================================
# YORDAMCHI FUNKSIYALAR
# =================================================================================
def load_g_dino_model():
    print(UZBEK_TEXT["loading_model"])
    g_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    print(UZBEK_TEXT["model_loaded"])
    return g_dino_model

def get_dominant_color(image, k=3):
    if image.size == 0: return ""
    pixels = image.reshape(-1, 3); pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color_bgr = centers[np.argmax(counts)]
    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    min_dist, closest_color_name = float('inf'), ""
    for color_name, (lower, upper) in UZBEK_TEXT["colors"].items():
        dist = min(abs(dominant_color_hsv[0] - lower[0]), abs(dominant_color_hsv[0] - upper[0]))
        if dist < min_dist: min_dist, closest_color_name = dist, color_name
    if dominant_color_hsv[2] > 200 and dominant_color_hsv[1] < 25: return "oq"
    if dominant_color_hsv[2] < 40: return "qora"
    return closest_color_name if min_dist < 40 else ""

def create_heatmap(background_image, points):
    height, width, _ = background_image.shape
    density_map = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        if 0 <= y < height and 0 <= x < width:
            density_map[y, x] += 1
    density_map = gaussian_filter(density_map, sigma=20)
    if np.max(density_map) > 0:
        density_map = (density_map / np.max(density_map) * 255)
    heatmap_color = cv2.applyColorMap(density_map.astype(np.uint8), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(background_image, 0.2, heatmap_color, 0.8, 0)
    cv2.imwrite("heatmap.jpg", superimposed_img)

def pixelate_face(face_roi):
    (h, w) = face_roi.shape[:2]
    w_px, h_px = max(1, w // PIXELATION_FACTOR), max(1, h // PIXELATION_FACTOR)
    temp = cv2.resize(face_roi, (w_px, h_px), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

# *** DINAMIK ZONALARNI YARATISH UCHUN YANGI FUNKSIYA ***
def create_dynamic_zones(shoe_detections, frame_width):
    if len(shoe_detections) == 0:
        return {}

    zones = {}
    frame_center_x = frame_width / 2
    
    left_shoes = shoe_detections[(shoe_detections.xyxy[:, 0] + shoe_detections.xyxy[:, 2]) / 2 < frame_center_x]
    right_shoes = shoe_detections[(shoe_detections.xyxy[:, 0] + shoe_detections.xyxy[:, 2]) / 2 >= frame_center_x]

    if len(left_shoes) > 0:
        x1 = np.min(left_shoes.xyxy[:, 0]) - ZONE_PADDING
        y1 = np.min(left_shoes.xyxy[:, 1]) - ZONE_PADDING
        x2 = np.max(left_shoes.xyxy[:, 2]) + ZONE_PADDING
        y2 = np.max(left_shoes.xyxy[:, 3]) + ZONE_PADDING
        zones["chap qator"] = {"coords": (int(x1), int(y1), int(x2), int(y2)), "color": (255, 165, 0)}

    if len(right_shoes) > 0:
        x1 = np.min(right_shoes.xyxy[:, 0]) - ZONE_PADDING
        y1 = np.min(right_shoes.xyxy[:, 1]) - ZONE_PADDING
        x2 = np.max(right_shoes.xyxy[:, 2]) + ZONE_PADDING
        y2 = np.max(right_shoes.xyxy[:, 3]) + ZONE_PADDING
        zones["o'ng qator"] = {"coords": (int(x1), int(y1), int(x2), int(y2)), "color": (0, 128, 255)}
        
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
        
    # Video writer setup
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
        
        enhanced_labels = []
        for bbox, class_id in zip(final_detections.xyxy, final_detections.class_id):
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = frame[y1:y2, x1:x2]
            color_name = get_dominant_color(cropped_image)
            current_class_name_en = CLASSES[class_id]
            current_class_name_uz = UZBEK_TEXT["class_names"].get(current_class_name_en, current_class_name_en)
            new_label = f"{current_class_name_uz} ({color_name})" if color_name else current_class_name_uz
            enhanced_labels.append(new_label)

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=final_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=final_detections, labels=enhanced_labels)
        
        try:
            faces = RetinaFace.detect_faces(annotated_frame)
            if isinstance(faces, dict):
                for face_key in faces.keys():
                    face_data = faces[face_key]
                    x1, y1, x2, y2 = face_data['facial_area']
                    face_roi = annotated_frame[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        pixelated_face = pixelate_face(face_roi)
                        annotated_frame[y1:y2, x1:x2] = pixelated_face
        except Exception as e:
            pass
            
        # *** DINAMIK ZONALARNI CHIZISH ***
        dynamic_zones = create_dynamic_zones(for_sale_shoe_detections, frame_width)
        for zone_name, zone_info in dynamic_zones.items():
            coords, color = zone_info["coords"], zone_info["color"]
            x1, y1, x2, y2 = coords
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, zone_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        men_count = sum(1 for cid in final_detections.class_id if CLASSES[cid] == "man")
        women_count = sum(1 for cid in final_detections.class_id if CLASSES[cid] == "woman")
        kids_count = sum(1 for cid in final_detections.class_id if CLASSES[cid] == "kid")
        shoes_count = sum(1 for cid in final_detections.class_id if CLASSES[cid] == "shoes")
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
        
        # Write frame to output video
        out.write(annotated_frame)

        
        display_frame = cv2.resize(annotated_frame, (1280, 720)) # You can adjust the size
        cv2.imshow("Space Analyzer - Press 'q' to quit", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        # Add heatmap points for later processing
        for bbox, class_id in zip(final_detections.xyxy, final_detections.class_id):
            if CLASSES[class_id] in ["man", "woman", "kid"]:
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                heatmap_points.append((center_x, center_y))

    cap.release()
    out.release()
    cv2.destroyAllWindows() # ADDED: Close the window at the end
    print(f"\nVideo saved as 'natija_test.mp4'")
    
    cap = cv2.VideoCapture("test.mp4")
    ret, first_frame = cap.read()
    if ret:
        create_heatmap(first_frame, heatmap_points)
        print(f"\n{UZBEK_TEXT['heatmap_saved']}")
    cap.release()

if __name__ == "__main__":
    main()
