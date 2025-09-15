import cv2
import torch
import numpy as np
import supervision as sv
from retinaface import RetinaFace
import sys
import re # Regular expressionlar uchun


# YANGI IMPORTLAR: Florence-2 uchun
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image


from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter


# =================================================================================
# O'ZBEK TILI UCHUN MATNLAR
# =================================================================================
UZBEK_TEXT = {
    "loading_model": "Florence-2 modeli yuklanmoqda...",
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
CLASSES = ["man", "woman", "kid", "shoes"]
DENSITY_THRESHOLDS = {"high": 0.5, "medium": 0.2}
STORE_AREA_SQ_METERS = 50.0
PIXELATION_FACTOR = 30
ZONE_PADDING = 20


# =================================================================================
# YORDAMCHI FUNKSIYALAR
# =================================================================================
def load_florence2_model():
    print(UZBEK_TEXT["loading_model"])
    model_id = 'microsoft/Florence-2-large'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, attn_implementation="eager").to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(UZBEK_TEXT["model_loaded"])
    return model, processor, device


def parse_florence2_output(parsed_output, class_map):
    if not isinstance(parsed_output, dict) or 'bboxes' not in parsed_output or 'labels' not in parsed_output:
        return sv.Detections.empty()

    boxes, class_ids = [], []
    for label, box in zip(parsed_output['labels'], parsed_output['bboxes']):
        if label == 'men':
            label = 'man'
        
        if label in class_map:
            boxes.append(box)
            class_ids.append(class_map[label])
            
    if not boxes:
        return sv.Detections.empty()
        
    return sv.Detections(xyxy=np.array(boxes), class_id=np.array(class_ids))


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


def create_dynamic_zones(shoe_detections, frame_width):
    if len(shoe_detections) == 0: return {}
    zones, frame_center_x = {}, frame_width / 2
    left_shoes = shoe_detections[(shoe_detections.xyxy[:, 0] + shoe_detections.xyxy[:, 2]) / 2 < frame_center_x]
    right_shoes = shoe_detections[(shoe_detections.xyxy[:, 0] + shoe_detections.xyxy[:, 2]) / 2 >= frame_center_x]
    if len(left_shoes) > 0:
        x1, y1, x2, y2 = np.min(left_shoes.xyxy[:, 0]), np.min(left_shoes.xyxy[:, 1]), np.max(left_shoes.xyxy[:, 2]), np.max(left_shoes.xyxy[:, 3])
        zones["chap qator"] = {"coords": (int(x1 - ZONE_PADDING), int(y1 - ZONE_PADDING), int(x2 + ZONE_PADDING), int(y2 + ZONE_PADDING)), "color": (255, 165, 0)}
    if len(right_shoes) > 0:
        x1, y1, x2, y2 = np.min(right_shoes.xyxy[:, 0]), np.min(right_shoes.xyxy[:, 1]), np.max(right_shoes.xyxy[:, 2]), np.max(right_shoes.xyxy[:, 3])
        zones["o'ng qator"] = {"coords": (int(x1 - ZONE_PADDING), int(y1 - ZONE_PADDING), int(x2 + ZONE_PADDING), int(y2 + ZONE_PADDING)), "color": (0, 128, 255)}
    return zones


# =================================================================================
# ASOSIY FUNKSIYA
# =================================================================================
def main():
    model, processor, device = load_florence2_model()
    
    cap = cv2.VideoCapture("test.mp4")
    if not cap.isOpened():
        print(UZBEK_TEXT["video_not_found"])
        return
        
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    heatmap_points = []
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # *** TUZATISH 1: Modelga aniqroq prompt berish ***
    # Modelga qidirilayotgan obyektlarni aniq aytamiz.
    task_prompt = '<OD>'
    prompt = f"{task_prompt} A photo of a {', '.join(CLASSES)}."
    
    class_map = {name: i for i, name in enumerate(CLASSES)}

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        print(UZBEK_TEXT["processing_frame"].format(frame_count), end='\r')

        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=prompt, images=image_pil, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                use_cache=False
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_text = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image_pil.width, image_pil.height))
        
        all_detections = parse_florence2_output(parsed_text.get(task_prompt, {}), class_map)

        if len(all_detections.xyxy) > 0:
            people_classes_en = {"man", "woman", "kid"}
            people_indices = [i for i, cid in enumerate(all_detections.class_id) if CLASSES[cid] in people_classes_en]
            shoe_indices = [i for i, cid in enumerate(all_detections.class_id) if CLASSES[cid] == "shoes"]
            
            worn_shoe_indices = set()
            if len(people_indices) > 0 and len(shoe_indices) > 0:
                for shoe_idx in shoe_indices:
                    shoe_box = all_detections.xyxy[shoe_idx]
                    shoe_center_x, shoe_center_y = (shoe_box[0] + shoe_box[2]) / 2, (shoe_box[1] + shoe_box[3]) / 2
                    for person_idx in people_indices:
                        person_box = all_detections.xyxy[person_idx]
                        if person_box[0] < shoe_center_x < person_box[2] and \
                           person_box[1] < shoe_center_y < person_box[3] and \
                           shoe_center_y > (person_box[1] + person_box[3]) / 2:
                            worn_shoe_indices.add(shoe_idx)
                            break
            
            for_sale_shoe_indices = [i for i in shoe_indices if i not in worn_shoe_indices]
            for_sale_shoe_detections = all_detections[for_sale_shoe_indices]
            final_indices = people_indices + for_sale_shoe_indices
            final_detections = all_detections[final_indices]
            
            enhanced_labels = []
            # *** TUZATISH 2: Detections bo'yicha to'g'ri iteratsiya qilish ***
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
            except Exception:
                pass
                
            dynamic_zones = create_dynamic_zones(for_sale_shoe_detections, frame_width)
            for zone_name, zone_info in dynamic_zones.items():
                coords, color = zone_info["coords"], zone_info["color"]
                x1, y1, x2, y2 = coords
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, zone_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            men_count = sum(1 for cid in final_detections.class_id if CLASSES[cid] == "man")
            women_count = sum(1 for cid in final_detections.class_id if CLASSES[cid] == "woman")
            kids_count = sum(1 for cid in final_detections.class_id if CLASSES[cid] == "kid")
            # *** TUZATISH 3: Faqat sotuvdagi poyabzallarni sanash ***
            shoes_count = len(for_sale_shoe_detections)
            total_people_count = men_count + women_count + kids_count
        else:
            # Agar hech qanday obyekt topilmasa
            annotated_frame = frame.copy()
            men_count, women_count, kids_count, shoes_count, total_people_count = 0, 0, 0, 0, 0

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
        
        cv2.imshow("Dynamic Store Analytics (Florence-2)", annotated_frame)
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
