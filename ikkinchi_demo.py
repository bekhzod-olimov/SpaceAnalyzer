import cv2
import torch
import numpy as np
import supervision as sv
import mediapipe as mp
from groundingdino.util.inference import Model
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter

# =================================================================================
# O'ZBEK TILI UCHUN MATNLAR
# =================================================================================
UZBEK_TEXT = {
    "loading_model": "Asosiy va yuzni aniqlash modellari yuklanmoqda...",
    "model_loaded": "Modellar muvaffaqiyatli yuklandi.",
    "video_not_found": "Xatolik: 'demo_video.mp4' video fayli topilmadi.",
    "processing_frame": "Kadr {} ishlanmoqda...",
    "detected_people": "Aniqlangan odamlar soni: {}",
    "detected_shoes": "Aniqlangan poyabzallar soni: {}",
    "density": "Zichlik: {:.2f} kishi/m²",
    "crowding_level": "Tirbandlik darajasi: {}",
    "low": "Past",
    "medium": "O'rtacha",
    "high": "Yuqori",
    "alert": "DIQQAT: '{}' hududida tirbandlik yuqori!",
    "heatmap_saved": "Faollik xaritasi 'heatmap_final.jpg' fayliga saqlandi.",
    "colors": {
        "qizil": ([0, 120, 70], [10, 255, 255]),
        "yashil": ([36, 100, 100], [86, 255, 255]),
        "ko'k": ([94, 80, 2], [126, 255, 255]),
        "qora": ([0, 0, 0], [180, 255, 30]),
        "oq": ([0, 0, 200], [180, 20, 255]),
        "kulrang": ([0, 0, 40], [180, 20, 200])
    }
}

# =================================================================================
# SOZLAMALAR
# =================================================================================
GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "weights/groundingdino_swint_ogc.pth"
CLASSES = ["person", "shoes"]
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
STORE_AREA_SQ_METERS = 50.0
DENSITY_THRESHOLDS = {"high": 0.5, "medium": 0.2}
HIGH_DENSITY_ALERT_ZONE = (100, 200, 600, 500)
BLUR_KERNEL_SIZE = (99, 99) # Yuzni xiralashtirish darajasi

# =================================================================================
# YORDAMCHI FUNKSIYALAR
# =================================================================================

def load_models():
    """Grounding DINO va MediaPipe modellarini yuklaydi."""
    print(UZBEK_TEXT["loading_model"])
    g_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    print(UZBEK_TEXT["model_loaded"])
    return g_dino_model, face_detector

def get_dominant_color(image, k=3):
    """Tasvirdagi ustun rangni aniqlaydi."""
    if image.size == 0: return ""
    pixels = image.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant_color_bgr = centers[np.argmax(counts)]
    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # Eng yaqin rang nomini topish
    min_dist, closest_color_name = float('inf'), ""
    for color_name, (lower, upper) in UZBEK_TEXT["colors"].items():
        dist = min(abs(dominant_color_hsv[0] - lower[0]), abs(dominant_color_hsv[0] - upper[0]))
        if dist < min_dist:
            min_dist, closest_color_name = dist, color_name
    
    if dominant_color_hsv[2] > 200 and dominant_color_hsv[1] < 25: return "oq"
    if dominant_color_hsv[2] < 40: return "qora"
    return closest_color_name if min_dist < 40 else ""

def create_heatmap(background_image, points):
    """Yakuniy faollik xaritasini yaratadi."""
    height, width, _ = background_image.shape
    density_map = np.zeros((height, width), dtype=np.float32)
    for x, y in points:
        if 0 <= y < height and 0 <= x < width:
            density_map[y, x] += 1
            
    density_map = gaussian_filter(density_map, sigma=16)
    if np.max(density_map) > 0:
        density_map = (density_map / np.max(density_map) * 255)
    
    heatmap_color = cv2.applyColorMap(density_map.astype(np.uint8), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(background_image, 0.5, heatmap_color, 0.5, 0)
    cv2.imwrite("heatmap_final.jpg", superimposed_img)

# =================================================================================
# ASOSIY FUNKSIYA
# =================================================================================

def main():
    # 1. Modellarni yuklash
    g_dino_model, face_detector = load_models()
    
    # 2. Videoni ochish
    cap = cv2.VideoCapture("demo_video.mp4")
    if not cap.isOpened():
        print(UZBEK_TEXT["video_not_found"])
        return
        
    # Annotatorlarni sozlash
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    heatmap_points = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        print(UZBEK_TEXT["processing_frame"].format(frame_count), end='\r')

        # 3. Yuzlarni aniqlash va xiralashtirish
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detector.process(rgb_frame)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    blurred_face = cv2.GaussianBlur(face_roi, BLUR_KERNEL_SIZE, 30)
                    frame[y:y+h, x:x+w] = blurred_face

        # 4. Asosiy ob'ektlarni aniqlash
        detections, labels = g_dino_model.predict_with_classes(
            image=frame, classes=CLASSES, box_threshold=BOX_TRESHOLD, text_threshold=TEXT_TRESHOLD
        )

        # 5. Rangni aniqlash va yorliqlarni yangilash
        enhanced_labels = []
        for i, (bbox, label) in enumerate(zip(detections.xyxy, labels)):
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = frame[y1:y2, x1:x2]
            color_name = get_dominant_color(cropped_image)
            new_label = f"{label.split(' ')[0]} ({color_name})" if color_name else label
            enhanced_labels.append(new_label)

        # 6. Annotatsiyalarni chizish
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=enhanced_labels)
        
        # 7. Statistikani hisoblash va ko'rsatish
        people_count = sum(1 for label in labels if "person" in label)
        shoes_count = sum(1 for label in labels if "shoes" in label)
        density = people_count / STORE_AREA_SQ_METERS

        crowding_level = UZBEK_TEXT["low"]
        if density >= DENSITY_THRESHOLDS["high"]: crowding_level = UZBEK_TEXT["high"]
        elif density >= DENSITY_THRESHOLDS["medium"]: crowding_level = UZBEK_TEXT["medium"]

        cv2.putText(annotated_frame, UZBEK_TEXT["detected_people"].format(people_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, UZBEK_TEXT["detected_shoes"].format(shoes_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, UZBEK_TEXT["density"].format(density), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, UZBEK_TEXT["crowding_level"].format(crowding_level), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 8. Signal (Alert) tizimi va Heatmap uchun nuqta yig'ish
        zx1, zy1, zx2, zy2 = HIGH_DENSITY_ALERT_ZONE
        cv2.rectangle(annotated_frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
        zone_people = 0
        
        for i, (detection, label) in enumerate(zip(detections.xyxy, labels)):
            if "person" in label:
                center_x, center_y = (detection[0] + detection[2]) / 2, (detection[1] + detection[3]) / 2
                heatmap_points.append((int(center_x), int(center_y)))
                if zx1 < center_x < zx2 and zy1 < center_y < zy2:
                    zone_people += 1
        
        zone_area = ((zx2 - zx1) * (zy2 - zy1)) / (frame.shape[1] * frame.shape[0]) * STORE_AREA_SQ_METERS
        zone_density = zone_people / zone_area if zone_area > 0 else 0
        if zone_density >= DENSITY_THRESHOLDS["high"]:
            alert_text = UZBEK_TEXT["alert"].format("Asosiy")
            cv2.putText(annotated_frame, alert_text, (zx1, zy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        # 9. Natijani ko'rsatish
        cv2.imshow("Full Store Analytics Demo", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 10. Heatmap yaratish
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        create_heatmap(first_frame, heatmap_points)
        print(f"\n{UZBEK_TEXT['heatmap_saved']}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

