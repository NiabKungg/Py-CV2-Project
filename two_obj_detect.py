import cv2
import os
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1

MAX_RESOLUTION = (1280, 720)  # 720p limit

def resize_frame(frame, max_resolution=MAX_RESOLUTION):
    h, w = frame.shape[:2]
    if h > max_resolution[1] or w > max_resolution[0]:
        scale = min(max_resolution[0] / w, max_resolution[1] / h)
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

# กำหนดสีแบบสุ่มสำหรับแต่ละประเภทของวัตถุ
CATEGORY_COLORS = {}

def get_color_for_category(category_name):
    """คืนค่าสีของ category_name ถ้ายังไม่มีใน dict ให้สร้างใหม่"""
    if category_name not in CATEGORY_COLORS:
        CATEGORY_COLORS[category_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return CATEGORY_COLORS[category_name]

def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and returns it."""
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (int(bbox.origin_x), int(bbox.origin_y))
        end_point = (int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height))
        
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability*100}%)"
        text_location = (MARGIN + int(bbox.origin_x), MARGIN + ROW_SIZE + int(bbox.origin_y))
        
        # ดึงสีของ category_name
        TEXT_COLOR = get_color_for_category(category_name)
        
        # วาดกรอบ
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # แสดงชื่อวัตถุ + ความน่าจะเป็น
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    
    return image

# Initialize the object detector
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.ObjectDetector.create_from_options(options)

def display_menu():
    menu_text = """

 $$$$$$\  $$\                                 $$\           $$$$$$$\             $$\                           $$\     $$\                     
$$  __$$\ $$ |                                $$ |          $$  __$$\            $$ |                          $$ |    \__|                    
$$ /  $$ |$$$$$$$\  $$\  $$$$$$\   $$$$$$$\ $$$$$$\         $$ |  $$ | $$$$$$\ $$$$$$\    $$$$$$\   $$$$$$$\ $$$$$$\   $$\  $$$$$$\  $$$$$$$\  
$$ |  $$ |$$  __$$\ \__|$$  __$$\ $$  _____|\_$$  _|        $$ |  $$ |$$  __$$\\_$$  _|  $$  __$$\ $$  _____|\_$$  _|  $$ |$$  __$$\ $$  __$$\ 
$$ |  $$ |$$ |  $$ |$$\ $$$$$$$$ |$$ /        $$ |          $$ |  $$ |$$$$$$$$ | $$ |    $$$$$$$$ |$$ /        $$ |    $$ |$$ /  $$ |$$ |  $$ |
$$ |  $$ |$$ |  $$ |$$ |$$   ____|$$ |        $$ |$$\       $$ |  $$ |$$   ____| $$ |$$\ $$   ____|$$ |        $$ |$$\ $$ |$$ |  $$ |$$ |  $$ |
 $$$$$$  |$$$$$$$  |$$ |\$$$$$$$\ \$$$$$$$\   \$$$$  |      $$$$$$$  |\$$$$$$$\  \$$$$  |\$$$$$$$\ \$$$$$$$\   \$$$$  |$$ |\$$$$$$  |$$ |  $$ |
 \______/ \_______/ $$ | \_______| \_______|   \____/       \_______/  \_______|  \____/  \_______| \_______|   \____/ \__| \______/ \__|  \__|
              $$\   $$ |                                                                                                                       
              \$$$$$$  |                                                                                                       [whit MediaPipe]
               \______/                                                                                                                        

╔═════════╦════════════════════════════════════════════════════════════════════════════════════╗
║ GitHub  ║ > https://github.com/NiabKungg                                                     ║
╠═════════╩════════════════════════════════════════════════════════════════════════════════════╣
║  Menu   ║  Select input source                                                               ║
║         ║  [1] Webcam                                                                        ║
║         ║  [2] Image/Video from 'sources' folder                                             ║
║         ║  [3] Back main memu                                                                ║
╚═════════╩════════════════════════════════════════════════════════════════════════════════════╝
"""
    os.system('cls' if os.name == 'nt' else 'clear')
    lines = menu_text.split("\n")
    num_lines = len(lines)
    for i, line in enumerate(lines):
        r = 0 #255 - int(i * (255 / num_lines))  # ไล่สีแดงจาก 255 -> 0
        g = 255 - int(i * (255 / num_lines))
        b = 0 #255 - int(i * (255 / num_lines))  # ไล่สีฟ้าจาก 255 -> 0 (เป็นสีม่วงไปขาว)
        gradient_color = f"\033[38;2;{r};{g};{b}m"  
        print(gradient_color + line + "\033[0m")  # รีเซ็ตสีหลังแต่ละบรรทัด
        

def display_file_menu(files):
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║           Select a File from 'sources' Folder          ║")
    print("╠════════════════════════════════════════════════════════╣")
    for i, file in enumerate(files):
        print(f"║ [{i+1}] {file.ljust(50)} ║")
    print("╚════════════════════════════════════════════════════════╝")

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_frame(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        annotated_image = visualize(rgb_frame, detection_result)
        output_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_image(file_path):
    image = cv2.imread(file_path)
    image = resize_frame(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect(mp_image)
    annotated_image = visualize(rgb_image, detection_result)
    output_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main program
os.system('cls' if os.name == 'nt' else 'clear')
display_menu()
choice = input("Enter choice (1 or 2): ")

if choice == "1":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        annotated_image = visualize(rgb_frame, detection_result)
        output_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Output", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

elif choice == "2":
    folder_path = "sources"
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.mp4', '.avi'))]
    
    if not files:
        print("No images or videos found in 'sources' folder.")
    else:
        display_file_menu(files)
        file_choice = int(input("Select a file number: ")) - 1
        if 0 <= file_choice < len(files):
            file_path = os.path.join(folder_path, files[file_choice])
            if file_path.endswith(('.jpg', '.png')):
                process_image(file_path)
            elif file_path.endswith(('.mp4', '.avi')):
                process_video(file_path)
        else:
            print("Invalid selection.")

elif choice == "3":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Back to main menu.")
    import main
    main.start()

else:
    print("Invalid choice. Exiting.")
