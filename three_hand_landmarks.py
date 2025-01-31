import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MAX_RESOLUTION = (1280, 720)  # 720p limit

def resize_frame(frame, max_resolution=MAX_RESOLUTION):
    h, w = frame.shape[:2]
    if h > max_resolution[1] or w > max_resolution[0]:
        scale = min(max_resolution[0] / w, max_resolution[1] / h)
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def draw_hand_skeleton(image, detection_result):
    image = image.copy()
    height, width, _ = image.shape

    for hand_landmarks in detection_result.hand_landmarks:
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            x1, y1 = int(start_point.x * width), int(start_point.y * height)
            x2, y2 = int(end_point.x * width), int(end_point.y * height)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for landmark in hand_landmarks:
            x, y = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    return image

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(image)

        if detection_result.hand_landmarks:
            frame = draw_hand_skeleton(frame, detection_result)
        
        frame = resize_frame(frame)
        cv2.imshow("Hand Skeleton - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_webcam():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = detector.detect(image)

        if detection_result.hand_landmarks:
            frame = draw_hand_skeleton(frame, detection_result)
        
        frame = resize_frame(frame)
        cv2.imshow("Hand Skeleton - Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def display_menu():
    menu_text = """
    
$$\   $$\                           $$\       $$\                           $$\                                   $$\                 
$$ |  $$ |                          $$ |      $$ |                          $$ |                                  $$ |                
$$ |  $$ | $$$$$$\  $$$$$$$\   $$$$$$$ |      $$ | $$$$$$\  $$$$$$$\   $$$$$$$ |$$$$$$\$$$$\   $$$$$$\   $$$$$$\  $$ |  $$\  $$$$$$$\ 
$$$$$$$$ | \____$$\ $$  __$$\ $$  __$$ |      $$ | \____$$\ $$  __$$\ $$  __$$ |$$  _$$  _$$\  \____$$\ $$  __$$\ $$ | $$  |$$  _____|
$$  __$$ | $$$$$$$ |$$ |  $$ |$$ /  $$ |      $$ | $$$$$$$ |$$ |  $$ |$$ /  $$ |$$ / $$ / $$ | $$$$$$$ |$$ |  \__|$$$$$$  / \$$$$$$\  
$$ |  $$ |$$  __$$ |$$ |  $$ |$$ |  $$ |      $$ |$$  __$$ |$$ |  $$ |$$ |  $$ |$$ | $$ | $$ |$$  __$$ |$$ |      $$  _$$<   \____$$\ 
$$ |  $$ |\$$$$$$$ |$$ |  $$ |\$$$$$$$ |      $$ |\$$$$$$$ |$$ |  $$ |\$$$$$$$ |$$ | $$ | $$ |\$$$$$$$ |$$ |      $$ | \$$\ $$$$$$$  |
\__|  \__| \_______|\__|  \__| \_______|      \__| \_______|\__|  \__| \_______|\__| \__| \__| \_______|\__|      \__|  \__|\_______/ 
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



# Main program
os.system('cls' if os.name == 'nt' else 'clear')
display_menu()
choice = input("Enter choice (1 or 2): ")

if choice == "1":
    process_webcam()
elif choice == "2":
    folder_path = "sources"
    files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.mp4', '.avi'))]

    display_file_menu(files)
    
    if not files:
        print("No images or videos found in 'sources' folder.")
        
    else:
        file_choice = int(input("Select a file number: ")) - 1
        if 0 <= file_choice < len(files):
            file_path = os.path.join(folder_path, files[file_choice])
            if file_path.endswith(('.mp4', '.avi')):
                process_video(file_path)
            else:
                frame = cv2.imread(file_path)
                frame = resize_frame(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection_result = detector.detect(image)
                
                if detection_result.hand_landmarks:
                    frame = draw_hand_skeleton(frame, detection_result)
                
                frame = resize_frame(frame)
                cv2.imshow("Hand Detection", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Invalid selection.")

elif choice == "3":
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Back to main menu.")
    import main
    main.start()

else:
    print("Invalid choice. Exiting.")

