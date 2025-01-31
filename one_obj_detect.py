import cv2
import os
import numpy as np
import random

MAX_RESOLUTION = (1280, 720)  # 720p limit

def resize_frame(frame, max_resolution=MAX_RESOLUTION):
    h, w = frame.shape[:2]
    if h > max_resolution[1] or w > max_resolution[0]:
        scale = min(max_resolution[0] / w, max_resolution[1] / h)
        new_size = (int(w * scale), int(h * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame

def get_color_for_category(category_idx):
    np.random.seed(category_idx)
    return tuple(np.random.randint(0, 255, 3).tolist())

# โหลดโมเดล MobileNet SSD
net = cv2.dnn.readNetFromCaffe('MobileNetSSD.prototxt', 'MobileNetSSD.caffemodel')

# รายชื่อคลาสที่โมเดลรู้จัก
Classes = ['Background', 'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair',
           'Cow', 'Diningtable', 'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa', 'Train', 'Tvmonitor']

def process_frame(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # กำหนด Threshold
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(Classes[idx], confidence * 100)
            color = get_color_for_category(idx)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow("Output", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    image = cv2.imread(image_path)
    image = resize_frame(image)
    processed_image = process_frame(image)
    cv2.imshow("Output", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
              \$$$$$$  |                                                                                                    [whit MobileNetSSD]
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
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow("Output", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

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