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
    "detected_people": "Aniqlangan odamlar soni: {}",
    "detected_shoes": "Aniqlangan poyabzallar soni: {}",
    "density": "Zichlik: {:.2f} kishi/metr kvadrat",
    "crowding_level": "Tirbandlik darajasi: {}",
    "low": "Past", "medium": "O'rtacha", "high": "Yuqori",
    "alert": "DIQQAT: '{}' hududida tirbandlik yuqori!",
    "heatmap_saved": "Faollik xaritasi 'heatmap.jpg' fayliga saqlandi.",
    "colors": {
        "qizil": ([0, 120, 70], [10, 255, 255]), "yashil": ([36, 100, 100], [86, 255, 255]),
        "ko'k": ([94, 80, 2], [126, 255, 255]), "qora": ([0, 0, 0], [180, 255, 30]),
        "oq": ([0, 0, 200], [180, 20, 255]), "kulrang": ([0, 0, 40], [180, 20, 200])
    }
}

# =================================================================================
# SOZLAMALAR
# =================================================================================
GROUNDING_DINO_CONFIG_PATH = "/home/bekhzod/Desktop/VideoDetection/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/bekhzod/Desktop/VideoDetection/GroundingDINO/weights/groundingdino_swint_ogc.pth"
CLASSES = ["person", "shoes"]
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
STORE_AREA_SQ_METERS = 50.0
DENSITY_THRESHOLDS = {"high": 0.5, "medium": 0.2}
HIGH_DENSITY_ALERT_ZONE = (100, 200, 600, 500)
PIXELATION_FACTOR = 30 

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
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color_bgr = centers[np.argmax(counts)]
    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    min_dist, closest_color_name = float('inf'), ""
    for color_name, (lower, upper) in UZBEK_TEXT["colors"].items():
        # *** XATO TO'G'RILANGAN QATOR ***
        dist = min(abs(dominant_color_hsv[0] - lower[0]), abs(dominant_color_hsv[0] - upper[0]))
        if dist < min_dist:
            min_dist, closest_color_name = dist, color_name
    
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

# =================================================================================
# ASOSIY FUNKSIYA
# =================================================================================
def main():
    g_dino_model = load_g_dino_model()
    cap = cv2.VideoCapture("test.mp4")
    if not cap.isOpened():
        print(UZBEK_TEXT["video_not_found"])
        return
        
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    heatmap_points = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        print(UZBEK_TEXT["processing_frame"].format(frame_count), end='\r')

        detections = g_dino_model.predict_with_classes(
            image=frame, classes=CLASSES, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD
        )

        enhanced_labels = []
        for bbox, class_id in zip(detections.xyxy, detections.class_id):
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = frame[y1:y2, x1:x2]
            color_name = get_dominant_color(cropped_image)
            current_class_name = "odam" if CLASSES[class_id] == "person" else "poyabzal"
            new_label = f"{current_class_name} ({color_name})" if color_name else current_class_name
            enhanced_labels.append(new_label)

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=enhanced_labels)
        
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
        
        people_count = sum(1 for cid in detections.class_id if CLASSES[cid] == "person")
        shoes_count = sum(1 for cid in detections.class_id if CLASSES[cid] == "shoes")
        density = people_count / STORE_AREA_SQ_METERS

        crowding_level = UZBEK_TEXT["low"]
        if density >= DENSITY_THRESHOLDS["high"]: crowding_level = UZBEK_TEXT["high"]
        elif density >= DENSITY_THRESHOLDS["medium"]: crowding_level = UZBEK_TEXT["medium"]

        rang = (0, 255, 0)
        cv2.putText(annotated_frame, UZBEK_TEXT["detected_people"].format(people_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, rang , 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, UZBEK_TEXT["detected_shoes"].format(shoes_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, rang, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, UZBEK_TEXT["density"].format(density), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, rang, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, UZBEK_TEXT["crowding_level"].format(crowding_level), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, rang, 2, cv2.LINE_AA)
        
        zx1, zy1, zx2, zy2 = HIGH_DENSITY_ALERT_ZONE
        cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
        zone_people = 0
        
        for i, (detection, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
            if CLASSES[class_id] == "person":
                center_x, center_y = (detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2
                heatmap_points.append((int(center_x), int(center_y)))
                if zx1 < center_x < zx2 and zy1 < center_y < zy2: zone_people += 1
        
        zone_area = ((zx2 - zx1) * (zy2 - zy1)) / (frame.shape[1] * frame.shape[0]) * STORE_AREA_SQ_METERS
        zone_density = zone_people / zone_area if zone_area > 0 else 0
        if zone_density >= DENSITY_THRESHOLDS["high"]:
            alert_text = UZBEK_TEXT["alert"].format("Asosiy")
            cv2.putText(annotated_frame, alert_text, (zx1, zy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("High-Accuracy Privacy Demo", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    
    cap = cv2.VideoCapture("test.mp4")
    ret, first_frame = cap.read()
    if ret:
        create_heatmap(first_frame, heatmap_points)
        print(f"\n{UZBEK_TEXT['heatmap_saved']}")
    cap.release()

if __name__ == "__main__":
    main()

