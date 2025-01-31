import cv2
import random
import os

face_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
color = [random.randint(0, 255) for _ in range(3)]
MAX_RESOLUTION = (1280, 720)  # 720p limit

# Resize function
def resize_frame(frame, max_resolution=MAX_RESOLUTION):
    h, w = frame.shape[:2]
    if h > max_resolution[1] or w > max_resolution[0]:
        scale = min(max_resolution[0] / w, max_resolution[1] / h)
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

def process_image(image_path="image.jpg"):
    """Process a static image."""
    image = cv2.imread(image_path)
    # Resize image
    image = resize_frame(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detect = face_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in face_detect:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(image, 'fullbody', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    cv2.imshow('Annotated Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_webcam():
    """Process live webcam feed."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = resize_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect = face_cascade.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in face_detect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, 'fullbody', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

        cv2.imshow('Webcam Output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_video(video_path="video.mp4"):
    """Process a video file for face detection."""
    if not os.path.exists(video_path):
        print(f"Error: '{video_path}' not found.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame = resize_frame(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect = face_cascade.detectMultiScale(gray, 1.1, 3)

        for (x, y, w, h) in face_detect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, 'fullbody', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

        cv2.imshow('Video Output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def display_menu():
    menu_text = """

$$$$$$$$\        $$\ $$\       $$$$$$$\                  $$\                 $$$$$$$\             $$\                           $$\     $$\                     
$$  _____|       $$ |$$ |      $$  __$$\                 $$ |                $$  __$$\            $$ |                          $$ |    \__|                    
$$ |   $$\   $$\ $$ |$$ |      $$ |  $$ | $$$$$$\   $$$$$$$ |$$\   $$\       $$ |  $$ | $$$$$$\ $$$$$$\    $$$$$$\   $$$$$$$\ $$$$$$\   $$\  $$$$$$\  $$$$$$$\  
$$$$$\ $$ |  $$ |$$ |$$ |      $$$$$$$\ |$$  __$$\ $$  __$$ |$$ |  $$ |      $$ |  $$ |$$  __$$\\_$$  _|  $$  __$$\ $$  _____|\_$$  _|  $$ |$$  __$$\ $$  __$$\ 
$$  __|$$ |  $$ |$$ |$$ |      $$  __$$\ $$ /  $$ |$$ /  $$ |$$ |  $$ |      $$ |  $$ |$$$$$$$$ | $$ |    $$$$$$$$ |$$ /        $$ |    $$ |$$ /  $$ |$$ |  $$ |
$$ |   $$ |  $$ |$$ |$$ |      $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |  $$ |      $$ |  $$ |$$   ____| $$ |$$\ $$   ____|$$ |        $$ |$$\ $$ |$$ |  $$ |$$ |  $$ |
$$ |   \$$$$$$  |$$ |$$ |      $$$$$$$  |\$$$$$$  |\$$$$$$$ |\$$$$$$$ |      $$$$$$$  |\$$$$$$$\  \$$$$  |\$$$$$$$\ \$$$$$$$\   \$$$$  |$$ |\$$$$$$  |$$ |  $$ |
\__|    \______/ \__|\__|      \_______/  \______/  \_______| \____$$ |      \_______/  \_______|  \____/  \_______| \_______|   \____/ \__| \______/ \__|  \__|
                                                             $$\   $$ |                                                                                         
                                                             \$$$$$$  |                                                                       [whit HaarCascade]
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