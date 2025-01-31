# Py-CV2 Project

โปรแกรมนี้ทำงานร่วมกับ OpenCV และ MediaPipe เพื่อให้คุณสามารถตรวจจับวัตถุต่างๆ และตรวจจับลักษณะเฉพาะของใบหน้าและร่างกายผ่านทางกล้องหรือไฟล์ภาพ/วิดีโอ

โปรแกรมนี้ประกอบด้วยเมนูการใช้งานต่างๆ สำหรับหลายๆ ฟีเจอร์:

1. การตรวจจับวัตถุ (MobileNetSSD / MediaPipe)
2. การตรวจจับ landmark ของมือ (MediaPipe)
3. การตรวจจับใบหน้าและลักษณะเฉพาะ (MediaPipe)
4. การตรวจจับดวงตา, รอยยิ้ม, และใบหน้าอื่นๆ โดยใช้ Haar Cascade

## ฟีเจอร์
- ตรวจจับวัตถุต่างๆ ด้วย MobileNetSSD และ MediaPipe
- ตรวจจับลักษณะของมือและใบหน้า
- ตรวจจับรอยยิ้มและดวงตา
- รองรับการใช้งานผ่านกล้องเว็บแคมหรือภาพ/วิดีโอ

## การตั้งค่า
โปรแกรมนี้ใช้ Python 3.x และไลบรารีที่จำเป็นดังนี้:
- OpenCV
- MediaPipe

### การติดตั้งไลบรารีที่จำเป็น
สามารถติดตั้งไลบรารีที่จำเป็นได้โดยใช้คำสั่ง:

```bash
pip install opencv-python mediapipe
```

## เมนูตัวเลือก
คุณสามารถเลือกตัวเลือกตามหมายเลขที่แสดงในเมนูเพื่อรันฟังก์ชันต่างๆ
- [1]: การตรวจจับวัตถุโดยใช้ MobileNetSSD
- [2]: การตรวจจับวัตถุโดยใช้ MediaPipe
- [3]: การตรวจจับ landmark ของมือ
- [4]: การตรวจจับใบหน้าด้วย MediaPipe
- [5]: การตรวจจับ landmark ของใบหน้า
- [6]: การตรวจจับ landmark ของร่างกาย
- [7]: การตรวจจับดวงตา (HaarCascade)
- [8]: การตรวจจับรอยยิ้ม (HaarCascade) (มีบั๊กบางประการ)
- [9]: การตรวจจับใบหน้าของแมว (HaarCascade)
- [10]: การตรวจจับใบหน้า (HaarCascade)
- [11]: การตรวจจับร่างกายทั้งหมด (HaarCascade)
- [12]: ออกจากโปรแกรม

## ตัวเลือกการรันฟังก์ชัน
โปรแกรมจะทำการเรียกใช้งานตามฟังก์ชันที่คุณเลือก และจะทำการแสดงผลลัพธ์การตรวจจับที่เกี่ยวข้องกับการเลือกของคุณ

# ตัวอย่างเช่น:
- เมื่อเลือกตัวเลือก [1]: โปรแกรมจะรันการตรวจจับวัตถุด้วย MobileNetSSD
- เมื่อเลือกตัวเลือก [4]: โปรแกรมจะรันการตรวจจับใบหน้าด้วย MediaPipe

## ไลบรารี
- Python 3.x
- OpenCV
- MediaPipe
