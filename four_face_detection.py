import cv2
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
MAX_RESOLUTION = (1280, 720)  # 720p limit

def resize_frame(frame, max_resolution=MAX_RESOLUTION):
    h, w = frame.shape[:2]
    if h > max_resolution[1] or w > max_resolution[0]:
        scale = min(max_resolution[0] / w, max_resolution[1] / h)
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

def _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height):
    """Converts normalized coordinates to pixel coordinates."""
    def is_valid(value):
        return 0 <= value <= 1
    if not (is_valid(normalized_x) and is_valid(normalized_y)):
        return None
    x_px = min(int(normalized_x * image_width), image_width - 1)
    y_px = min(int(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def visualize(image, detection_result):
    """Draws bounding boxes and keypoints on the image."""
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
            if keypoint_px:
                cv2.circle(annotated_image, keypoint_px, 2, (0, 255, 0), -1)

        category = detection.categories[0]
        text = f"{category.category_name} ({round(category.score, 2)})"
        text_location = (bbox.origin_x + MARGIN, bbox.origin_y + ROW_SIZE + MARGIN)
        cv2.putText(annotated_image, text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image

# Load face detector
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

def process_image(image_path="image.jpg"):
    """Process a static image with resizing."""
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Cannot read image file")
        return

    frame = resize_frame(frame)  # ✅ Resize ภาพก่อนประมวลผล

    # แปลงเป็น Mediapipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)

    # วาดผลลัพธ์ลงบนภาพที่ resize แล้ว
    annotated_image = visualize(frame.copy(), detection_result)

    # แสดงผลภาพ
    cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    

def process_webcam():
    """Process frames from webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = resize_frame(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        annotated_frame = visualize(frame, detection_result)

        cv2.imshow("Webcam - Face Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

import time

def process_video(video_path="video.mp4"):
    """Process a video file and apply face detection at normal speed."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get FPS from video
    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30  # Expected time per frame

    while cap.isOpened():
        start_time = time.time()  # Record frame start time
        
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame = resize_frame(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        annotated_frame = visualize(frame, detection_result)

        cv2.imshow("Video - Face Detection", annotated_frame)

        elapsed_time = time.time() - start_time  # Time taken to process frame
        sleep_time = frame_time - elapsed_time  # Adjust for real-time playback

        if sleep_time > 0:
            time.sleep(sleep_time)  # Pause if frame processed too quickly

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()


def display_menu():
    menu_text = """

$$$$$$$$\                                       $$\            $$\                           $$\     $$\                     
$$  _____|                                      $$ |           $$ |                          $$ |    \__|                    
$$ |   $$$$$$\   $$$$$$$\  $$$$$$\         $$$$$$$ | $$$$$$\ $$$$$$\    $$$$$$\   $$$$$$$\ $$$$$$\   $$\  $$$$$$\  $$$$$$$\  
$$$$$\ \____$$\ $$  _____|$$  __$$\       $$  __$$ |$$  __$$\\_$$  _|  $$  __$$\ $$  _____|\_$$  _|  $$ |$$  __$$\ $$  __$$\ 
$$  __|$$$$$$$ |$$ /      $$$$$$$$ |      $$ /  $$ |$$$$$$$$ | $$ |    $$$$$$$$ |$$ /        $$ |    $$ |$$ /  $$ |$$ |  $$ |
$$ |  $$  __$$ |$$ |      $$   ____|      $$ |  $$ |$$   ____| $$ |$$\ $$   ____|$$ |        $$ |$$\ $$ |$$ |  $$ |$$ |  $$ |
$$ |  \$$$$$$$ |\$$$$$$$\ \$$$$$$$\       \$$$$$$$ |\$$$$$$$\  \$$$$  |\$$$$$$$\ \$$$$$$$\   \$$$$  |$$ |\$$$$$$  |$$ |  $$ |
\__|   \_______| \_______| \_______|       \_______| \_______|  \____/  \_______| \_______|   \____/ \__| \______/ \__|  \__|
                                                                                                             [whit MediaPipe]

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
        r = 255 - int(i * (255 / num_lines))  # ไล่สีแดงจาก 255 -> 0
        g = 0 #255 - int(i * (255 / num_lines))
        b = 0 #255 - int(i * (255 / num_lines))  # ไล่สีฟ้าจาก 255 -> 0 (เป็นสีม่วงไปขาว)
        gradient_color = f"\033[38;2;{r};{g};{b}m"  
        print(gradient_color + line + "\033[0m")  # รีเซ็ตสีหลังแต่ละบรรทัด

def main_display_menu():
    display_menu()
    return input("เลือกโหมดการทำงาน (1 หรือ 2): ")

def display_file_menu(files):
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║           Select a File from 'sources' Folder          ║")
    print("╠════════════════════════════════════════════════════════╣")
    for i, file in enumerate(files):
        print(f"║ [{i+1}] {file.ljust(50)} ║")
    print("╚════════════════════════════════════════════════════════╝")

def main_display_file_menu(files):
    display_file_menu(files)
    return int(input("เลือกไฟล์หมายเลข: ")) - 1

choice = main_display_menu()
if choice == "1":
    process_webcam()

elif choice == "2":
    folder_path = "sources"
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.mp4', '.avi'))]
    if not files:
        print("ไม่มีไฟล์ในโฟลเดอร์ 'sources'")
    else:
        file_choice = main_display_file_menu(files)
        file_path = os.path.join(folder_path, files[file_choice])
        if file_path.endswith(('.jpg', '.png')):
            process_image(file_path)
        elif file_path.endswith(('.mp4', '.avi')):
            process_video(file_path)

elif choice == "3":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Back to main menu.")
    import main
    main.start()

else:
    print("Invalid choice. Exiting.")