import os
import time

menu_text = """

$$$$$$$\                     $$$$$$\  $$\    $$\  $$$$$$\        $$$$$$$\                                                $$\     
$$  __$$\                   $$  __$$\ $$ |   $$ |$$  __$$\       $$  __$$\                                               $$ |    
$$ |  $$ |$$\   $$\         $$ /  \__|$$ |   $$ |\__/  $$ |      $$ |  $$ | $$$$$$\   $$$$$$\  $$\  $$$$$$\   $$$$$$$\ $$$$$$\   
$$$$$$$  |$$ |  $$ |$$$$$$\ $$ |      \$$\  $$  | $$$$$$  |      $$$$$$$  |$$  __$$\ $$  __$$\ \__|$$  __$$\ $$  _____|\_$$  _|  
$$  ____/ $$ |  $$ |\______|$$ |       \$$\$$  / $$  ____/       $$  ____/ $$ |  \__|$$ /  $$ |$$\ $$$$$$$$ |$$ /        $$ |    
$$ |      $$ |  $$ |        $$ |  $$\   \$$$  /  $$ |            $$ |      $$ |      $$ |  $$ |$$ |$$   ____|$$ |        $$ |$$\ 
$$ |      \$$$$$$$ |        \$$$$$$  |   \$  /   $$$$$$$$\       $$ |      $$ |      \$$$$$$  |$$ |\$$$$$$$\ \$$$$$$$\   \$$$$  |
\__|       \____$$ |         \______/     \_/    \________|      \__|      \__|       \______/ $$ | \_______| \_______|   \____/ 
          $$\   $$ |                                                                     $$\   $$ |                        [v1.0]
          \$$$$$$  |                                                                     \$$$$$$  |                     [By Niab]
           \______/                                                                       \______/                               

╔═════════╦════════════════════════════════════════════════════════════════════════════════════╗
║ GitHub  ║ > https://github.com/NiabKungg                                                     ║
╠═════════╩════════════════════════════════════════════════════════════════════════════════════╣
║  Menu   ║  [1] Object Detection - (MobileNetSSD)                                             ║
║         ║  [2] Object Detection - (MediaPipe)                                                ║
║         ║  [3] Hand Landmark Detection - (MediaPipe)                                         ║
║         ║  [4] Face Detection - (MediaPipe)                                                  ║
║         ║  [5] Face Landmark Detection - (MediaPipe)                                         ║
║         ║  [6] Pose Landmark Detection - (MediaPipe)                                         ║
║         ║  [7] Eyes Detection - (HaarCascade)                                                ║
║         ║  [8] Smile Detection - (HaarCascade) (Lots of bugs)                                ║
║         ║  [9] Frontal Cat Face Detection - (HaarCascade)                                    ║
║         ║  [10] Frontal Face Detection - (HaarCascade)                                       ║
║         ║  [11] Full Body Detection - (HaarCascade)                                          ║
║         ║  [12] Exit                                                                         ║
╚═════════╩════════════════════════════════════════════════════════════════════════════════════╝
"""

def start(status=""):
    os.system('cls' if os.name == 'nt' else 'clear')
    lines = menu_text.split("\n")
    num_lines = len(lines)
    for i, line in enumerate(lines):
        r = 0 #255 - int(i * (255 / num_lines))  # ไล่สีแดงจาก 255 -> 0
        g = 0
        b = 255 - int(i * (255 / num_lines))  # ไล่สีฟ้าจาก 255 -> 0 (เป็นสีม่วงไปขาว)
        gradient_color = f"\033[38;2;{r};{g};{b}m"  
        print(gradient_color + line + "\033[0m")  # รีเซ็ตสีหลังแต่ละบรรทัด

    if status != "":
        print(status)
        print()

    while True:
        choice = input("Enter your choice: ")

        if choice == "1":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Object Detection with MobileNetSSD...\033[0m')
            time.sleep(0.5)
            import one_obj_detect
            exit()

        elif choice == "2":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Object Detection with MediaPipe...\033[0m')
            time.sleep(0.5)
            import two_obj_detect
            exit()

        elif choice == "3":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Hand Landmark Detection whit MediaPipe\033[0m')
            time.sleep(0.5)
            import three_hand_landmarks
            exit()

        elif choice == "4":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Face Detection with MediaPipe...\033[0m')
            time.sleep(0.5)
            import four_face_detection
            exit()

        elif choice == "5":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Face Landmark Detection with MediaPipe...\033[0m')
            time.sleep(0.5)
            import five_face_landmark
            exit()

        elif choice == "6":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Pose Landmark Detection with MediaPipe...\033[0m')
            time.sleep(0.5)
            import six_pose_landmark
            exit()

        elif choice == "7":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Eyes Detection with HaarCascade...\033[0m') 
            time.sleep(0.5)
            import seven_eyes_detection
            exit()

        elif choice == "8":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Smile Detection with HaarCascade...\033[0m')
            time.sleep(0.5)
            import eight_smile_detection
            exit()

        elif choice == "9":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Frontal Cat Face Detection with HaarCascade...\033[0m')
            time.sleep(0.5)
            import nine_frontal_cat_face_detection
            exit()

        elif choice == "10":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Frontal Face Detection with HaarCascade...\033[0m')
            time.sleep(0.5)
            import ten_frontal_face_detection

        elif choice == "11":
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\033[32mRunning Full Body Detection with HaarCascade...\033[0m')
            time.sleep(0.5)
            import eleven_full_body_detection

        elif choice == "12":
            exit()
            
        else:
            os.system('cls' if os.name == 'nt' else 'clear')
            start("\033[31mInvalid choice. Please try again.\033[0m")
            exit()

start()