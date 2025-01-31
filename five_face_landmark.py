import cv2
import time
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

MAX_RESOLUTION = (1280, 720)  # 720p limit

def resize_frame(frame, max_resolution=MAX_RESOLUTION):
    """ปรับขนาดเฟรมให้ไม่เกินความละเอียดสูงสุด"""
    h, w = frame.shape[:2]
    if h > max_resolution[1] or w > max_resolution[0]:
        scale = min(max_resolution[0] / w, max_resolution[1] / h)
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

def draw_landmarks_on_image(rgb_image, detection_result):
    """วาดจุด Landmark บนภาพที่ตรวจจับได้"""
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image

# สร้าง FaceLandmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def process_image(image_path="image.png"):
    """ประมวลผลภาพจากไฟล์"""
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Cannot read image file")
        return

    frame = resize_frame(frame)  # ✅ ใช้ resize_frame ก่อนประมวลผล

    # แปลงเป็น Mediapipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(mp_image)

    annotated_image = draw_landmarks_on_image(frame, detection_result)
    
    # แสดงผลภาพ
    cv2.imshow("Face Landmark Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_webcam():
    """ประมวลผลวิดีโอจากเว็บแคม"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_frame(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        annotated_frame = draw_landmarks_on_image(frame, detection_result)

        cv2.imshow("Webcam - Face Landmark Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # กด 'q' เพื่อออก
            break

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path="video.mp4"):
    """ประมวลผลวิดีโอจากไฟล์"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30  # ปรับตาม FPS ของวิดีโอ

    while cap.isOpened():
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_frame(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)

        annotated_frame = draw_landmarks_on_image(frame, detection_result)

        cv2.imshow("Video - Face Landmark Detection", annotated_frame)

        elapsed_time = time.time() - start_time
        sleep_time = frame_time - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # กด 'q' เพื่อออก
            break

    cap.release()
    cv2.destroyAllWindows()


def display_menu():
    menu_text = """

$$$$$$$$\                                 $$\                                $$\                                   $$\       
$$  _____|                                $$ |                               $$ |                                  $$ |      
$$ |   $$$$$$\   $$$$$$$\  $$$$$$\        $$ |      $$$$$$\  $$$$$$$\   $$$$$$$ |$$$$$$\$$$$\   $$$$$$\   $$$$$$\  $$ |  $$\ 
$$$$$\ \____$$\ $$  _____|$$  __$$\       $$ |      \____$$\ $$  __$$\ $$  __$$ |$$  _$$  _$$\  \____$$\ $$  __$$\ $$ | $$  |
$$  __|$$$$$$$ |$$ /      $$$$$$$$ |      $$ |      $$$$$$$ |$$ |  $$ |$$ /  $$ |$$ / $$ / $$ | $$$$$$$ |$$ |  \__|$$$$$$  / 
$$ |  $$  __$$ |$$ |      $$   ____|      $$ |     $$  __$$ |$$ |  $$ |$$ |  $$ |$$ | $$ | $$ |$$  __$$ |$$ |      $$  _$$<  
$$ |  \$$$$$$$ |\$$$$$$$\ \$$$$$$$\       $$$$$$$$\\$$$$$$$ |$$ |  $$ |\$$$$$$$ |$$ | $$ | $$ |\$$$$$$$ |$$ |      $$ | \$$\ 
\__|   \_______| \_______| \_______|      \________|\_______|\__|  \__| \_______|\__| \__| \__| \_______|\__|      \__|  \__|
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