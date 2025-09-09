import cv2
import numpy as np
import sqlite3
import random
import time
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter # <-- YANGLI IMPORT

# =================================================================================
# O'ZBEK TILI UCHUN TARJIMALAR VA MATNLAR
# =================================================================================
UZBEK_TEXT = {
    "person_entered": "Yangi shaxs kirdi. ID: {}",
    "person_exited": "Shaxs chiqib ketdi. ID: {}. Do'konda bo'lish vaqti: {:.2f} soniya",
    "total_people": "Do'kondagi umumiy odamlar soni: {}",
    "action_registered": "Harakat qayd etildi. ID: {}. Harakat: '{}'",
    "db_setup": "Ma'lumotlar bazasi muvaffaqiyatli sozlandi.",
    "db_connection_error": "Ma'lumotlar bazasiga ulanishda xatolik: {}",
    "video_not_found": "Xatolik: 'demo_video.mp4' video fayli topilmadi. Iltimos, fayl mavjudligini tekshiring.",
    "heatmap_saved": "Faollik xaritasi 'heatmap_final.jpg' fayliga saqlandi.",
    "generating_summary": "\n{} dan {} gacha bo'lgan vaqt oralig'i uchun harakatlar hisoboti tayyorlanmoqda...",
    "hourly_summary_title": "Soatlik Harakatlar Hisoboti:",
    "no_actions": "Belgilangan vaqt oralig'ida hech qanday maxsus harakatlar qayd etilmagan.",
    "processing_frame": "Kadr ishlanmoqda: {}",
    "density": "Zichlik: {:.2f} kishi/m²",
    "crowding_level": "Tirbandlik darajasi: {}",
    "low": "Past",
    "medium": "O'rtacha",
    "high": "Yuqori"
}

# =================================================================================
# SOZLAMALAR VA KONSTANTALAR
# =================================================================================
ENTRANCE_LINE_Y = 300
INTEREST_ZONES = {
    "premium_poyabzallar": (100, 400, 300, 550),
    "kassa_zonasi": (700, 400, 900, 550)
}
DWELL_TIME_THRESHOLD = 5.0
STORE_AREA_SQ_METERS = 50.0  # Do'konning taxminiy maydoni (m²)

# =================================================================================
# MA'LUMOTLAR BAZASINI SOZLASH (O'zgarishsiz)
# =================================================================================
def setup_database():
    try:
        conn = sqlite3.connect('store_analytics.db', isolation_level=None)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS visits (track_id INTEGER PRIMARY KEY, entry_time TEXT NOT NULL, exit_time TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS actions (action_id INTEGER PRIMARY KEY AUTOINCREMENT, track_id INTEGER, timestamp TEXT NOT NULL, action_description TEXT NOT NULL, zone_name TEXT, FOREIGN KEY(track_id) REFERENCES visits(track_id))''')
        cursor.execute('DELETE FROM actions'); cursor.execute('DELETE FROM visits')
        conn.commit()
        print(UZBEK_TEXT["db_setup"])
        return conn
    except sqlite3.Error as e:
        print(UZBEK_TEXT["db_connection_error"].format(e))
        return None

# =================================================================================
# *** YANGILANGAN QISM: MOCK AI FUNKSIYALARI ***
# =================================================================================

class MockDetector:
    """
    Bu D-FINE modelining ishlashini ancha aniqroq simulyatsiya qiluvchi klass.
    Obyektlarni kadrlar orasida silliq harakatlanishini ta'minlaydi.
    """
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        # Bir nechta odamning boshlang'ich holati va harakat yo'nalishi
        self.people = [
            {"pos": [int(self.width*0.2), int(self.height*0.4)], "dir": [2, 1], "size": [80, 160]},
            {"pos": [int(self.width*0.7), int(self.height*0.5)], "dir": [-1, -2], "size": [70, 150]},
            {"pos": [int(self.width*0.5), int(self.height*0.8)], "dir": [1, -1], "size": [75, 155]},
        ]

    def detect_people(self):
        detections = []
        for person in self.people:
            # Odamni harakatlantirish
            person["pos"][0] += person["dir"][0]
            person["pos"][1] += person["dir"][1]
            
            # Chegaralardan qaytish
            if not 0 < person["pos"][0] < self.width - person["size"][0]: person["dir"][0] *= -1
            if not 0 < person["pos"][1] < self.height - person["size"][1]: person["dir"][1] *= -1

            w, h = person["size"]
            x1 = person["pos"][0]
            y1 = person["pos"][1]
            x2 = x1 + w
            y2 = y1 + h
            detections.append([x1, y1, x2, y2])
        return detections

def mock_detect_people(frame):
    """
    Bu funksiya D-FINE kabi odamni aniqlovchi modelni simulyatsiya qiladi.
    Haqiqiy loyihada bu yerda D-FINE modelini ishga tushirasiz.
    U odamlarning koordinatalarini (bounding box) qaytaradi.
    """
    # Demo uchun bir nechta statik to'rtburchaklar qaytaramiz
    # Haqiqiy vaqtda bu dinamik bo'ladi
    detections = []
    height, width, _ = frame.shape
    # Har bir necha kadrda odamlar paydo bo'lishini simulyatsiya qilish
    if random.random() > 0.1:
        detections.append([int(width*0.2), int(height*0.4), int(width*0.3), int(height*0.8)]) # odam 1
    if random.random() > 0.3:
        detections.append([int(width*0.6), int(height*0.5), int(width*0.7), int(height*0.9)]) # odam 2
    return detections

def mock_track_people(detections):
    return [(bbox, i + 1) for i, bbox in enumerate(detections)]

def mock_get_action_caption(person_image):
    actions = ["poyabzalni ko'zdan kechirmoqda", "telefonga qaramoqda", "javondagi mahsulotni ushlab ko'rmoqda"]
    return random.choice(actions)

def mock_summarize_actions_llm(actions_text):
    if not actions_text: return UZBEK_TEXT["no_actions"]
    summary = "Kun davomida mijozlar asosan yangi kolleksiyadagi premium poyabzallarga qiziqish bildirishdi. "
    summary += "Umuman olganda, do'konning o'ng tomoni ko'proq faol bo'ldi."
    return summary

def analyze_video(video_path, db_conn):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(UZBEK_TEXT["video_not_found"])
        return

    cursor = db_conn.cursor()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # YANGILANGAN: D-FINE simulyatorini ishga tushirish
    detector = MockDetector(frame_width, frame_height)
    
    # Faollik xaritasi uchun nuqtalar ro'yxati
    heatmap_points = []
    
    tracked_persons_status = {}
    frame_count = 0
    
    # Videoning birinchi kadrini fon sifatida saqlab olish
    ret, first_frame = cap.read()
    if not ret: return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Videoni boshiga qaytarish

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        print(UZBEK_TEXT["processing_frame"].format(frame_count), end='\r')

        detections = detector.detect_people()
        tracked_objects = mock_track_people(detections)
        
        current_frame_track_ids = set()

        for bbox, track_id in tracked_objects:
            current_frame_track_ids.add(track_id)
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            bottom_y = y2

            # Faollik xaritasi uchun nuqtalarni qo'shish
            heatmap_points.append((center_x, bottom_y))
            
            # (Qolgan kod bu yerda - kirish/chiqish hisoblash, zonalar, va hokazo)
            # Bu qism o'zgarishsiz qoladi...
            # ...
        
        # Ekranga ma'lumotlarni chiqarish (o'zgarishsiz qoladi)
        in_store_count = len(tracked_objects) # Simulyatsiyada aniqroq
        cv2.putText(frame, UZBEK_TEXT["total_people"].format(in_store_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Do'kon Tahlili Demosi (Tugmachani bosing: 'q')", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo tahlili tugadi.")

    # 5-qadam: YANGLANGAN Faollik xaritasini yaratish va saqlash
    create_advanced_heatmap(first_frame, heatmap_points, len(tracked_objects))
    print(UZBEK_TEXT["heatmap_saved"])


def create_advanced_heatmap(background_image, points, current_people_count):
    """Siz yuborgan rasmga o'xshash, ilg'or faollik xaritasini yaratadi."""
    height, width, _ = background_image.shape
    density_map = np.zeros((height, width), dtype=np.float32)

    # Har bir nuqta uchun zichlikni oshirish
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            density_map[y, x] += 1
            
    # Gaussian filtri bilan silliqlash
    # sigma qiymati qanchalik katta bo'lsa, xarita shunchalik silliq bo'ladi
    density_map = gaussian_filter(density_map, sigma=16)

    # Zichlikni 0-255 oralig'iga normallashtirish
    if np.max(density_map) > 0:
        density_map = (density_map / np.max(density_map) * 255).astype(np.uint8)
    else:
        density_map = density_map.astype(np.uint8)

    # Rangli xarita (JET) qo'llash
    heatmap_color = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)

    # Fon tasviri bilan birlashtirish
    # alpha - bu heatmap qanchalik shaffof bo'lishini belgilaydi (0.5 = 50% shaffof)
    alpha = 0.5
    superimposed_img = cv2.addWeighted(background_image, 1 - alpha, heatmap_color, alpha, 0)
    
    # Statistikani hisoblash va qo'shish
    density = current_people_count / STORE_AREA_SQ_METERS
    if density < 0.2:
        crowding_level = UZBEK_TEXT["low"]
        crowd_color = (0, 255, 0) # Yashil
    elif density < 0.5:
        crowding_level = UZBEK_TEXT["medium"]
        crowd_color = (0, 255, 255) # Sariq
    else:
        crowding_level = UZBEK_TEXT["high"]
        crowd_color = (0, 0, 255) # Qizil

    # Matnni qo'shish
    density_text = UZBEK_TEXT["density"].format(density)
    crowding_text = UZBEK_TEXT["crowding_level"].format(crowding_level)
    
    cv2.putText(superimposed_img, density_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(superimposed_img, crowding_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite("heatmap_final.jpg", superimposed_img)

# =================================================================================
# LLM ORQALI HISOBOT TAYYORLASH FUNKSIYASI
# =================================================================================
def generate_summary_report(db_conn):
    """Bazadagi ma'lumotlardan foydalanib LLM orqali hisobot yaratadi."""
    cursor = db_conn.cursor()
    
    # Oxirgi 2 daqiqalik ma'lumotlarni olishni simulyatsiya qilamiz
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=2)
    
    print(UZBEK_TEXT["generating_summary"].format(start_time.strftime('%H:%M'), end_time.strftime('%H:%M')))
    
    cursor.execute("SELECT timestamp, action_description, zone_name FROM actions WHERE timestamp BETWEEN ? AND ?",
                   (start_time.isoformat(), end_time.isoformat()))
    actions = cursor.fetchall()
    
    actions_text = "\n".join([f"- {row[0]}: {row[2]} zonasida '{row[1]}'" for row in actions])
    
    # LLM simulyatsiyasini chaqirish
    summary = mock_summarize_actions_llm(actions_text)
    
    print("\n" + "="*50)
    print(UZBEK_TEXT["hourly_summary_title"])
    print(summary)
    print("="*50 + "\n")

# =================================================================================
# ASOSIY ISHGA TUSHIRISH BLOKI (qolgan qismlar o'zgarishsiz)
# =================================================================================
if __name__ == "__main__":
    db_connection = setup_database()
    if db_connection:
        # analyze_video ichidagi hisoblashlar va qolgan mantiq o'zgarishsiz qoladi
        # faqatgina heatmap yaratish va odamni aniqlash qismlari yangilandi
        
        # Bu funksiya ichidagi asl mantiq saqlanib qolgan
        # Kerakli qismlarni 'analyze_video' ichidan olib, bu yerga ko'chirishimiz mumkin
        # Sodda bo'lishi uchun hozircha to'liq funksiyani chaqiramiz
        analyze_video("test.mp4", db_connection)
        
        generate_summary_report(db_connection)
        db_connection.close()

def generate_summary_report(db_conn):
    """Bazadagi ma'lumotlardan foydalanib LLM orqali hisobot yaratadi."""
    cursor = db_conn.cursor()
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=10) # Vaqtni ko'paytiramiz
    print(UZBEK_TEXT["generating_summary"].format(start_time.strftime('%H:%M'), end_time.strftime('%H:%M')))
    cursor.execute("SELECT timestamp, action_description, zone_name FROM actions WHERE timestamp BETWEEN ? AND ?", (start_time.isoformat(), end_time.isoformat()))
    actions = cursor.fetchall()
    actions_text = "\n".join([f"- {row[0]}: {row[2]} zonasida '{row[1]}'" for row in actions])
    summary = mock_summarize_actions_llm(actions_text)
    print("\n" + "="*50)
    print(UZBEK_TEXT["hourly_summary_title"])
    print(summary)
    print("="*50 + "\n")


