# 🚦 Traffic Violation Detection System (Red Light Jump Detection)

An **AI-powered Traffic Violation Detection System** built using **Python, OpenCV, Flask, and OCR** that automatically detects **red light violations**, extracts **vehicle license plate numbers**, stores **violation records**, and generates a **processed evidence video**.

This project is suitable for **smart traffic surveillance**, **law enforcement automation**, and **computer vision–based academic or industry projects**.

---

## 📌 Features

- 🎥 Upload traffic surveillance videos  
- 🚦 Automatic traffic signal color detection (Red / Yellow / Green)  
- ➖ White stop-line detection using Hough Transform  
- 🚗 Detect vehicles crossing the stop-line during **RED** signal  
- 🔍 License plate extraction using contour detection  
- 🧠 OCR-based number plate recognition  
  - Tesseract OCR  
  - EasyOCR (fallback)  
- 📸 Saves screenshot evidence of violating number plates  
- 📊 Stores violation count, timestamps, and images  
- 📈 Live processing status & progress tracking  
- 📥 Download processed violation video  
- 🌐 REST APIs for violations data  
- 🧹 Clear all stored violations with one click  
- ⚡ Optimized processing using frame skipping  

---

## 🛠 Tech Stack

### Backend
- Python  
- Flask  
- OpenCV  
- NumPy  
- Threading & Queue  

### OCR
- Tesseract OCR  
- EasyOCR  

### Frontend
- HTML  
- CSS  
- JavaScript  
- Jinja2 Templates  

### Storage
- In-memory data structures  
- Local image storage  

---

## 📂 Project Structure

```bash
traffic-violation-detection/
│
├── app.py                     # Main Flask application
├── uploads/                    # Uploaded videos
├── processed/                  # Processed output videos
├── violations/                 # Captured license plate images
│
├── templates/
│   ├── index.html
│   ├── upload.html
│   ├── processing.html
│   ├── results.html
│   └── violations.html
│
├── static/
│   ├── css/
│   └── js/
│
├── traffic_video.mp4           # Demo video (optional)
├── requirements.txt
└── README.md
⚙️ Installation & Setup
1️⃣ Clone the Repository

```
git clone https://github.com/your-username/traffic-violation-detection.git
cd traffic-violation-detection
```
2️⃣ Create Virtual Environment (Recommended)
```
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```
OCR Setup
✅ Tesseract OCR

Download from:
https://github.com/tesseract-ocr/tesseract

Update path in code (already included):
```
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

✅ EasyOCR

Automatically used as a fallback if Tesseract fails.

▶️ Run the Application

```
python app.py

```

Server runs at: http://localhost:5000

```

## 🔄 Working Logic (Pipeline)
```
Video upload
Frame sampling (every 10th frame)
Traffic signal color detection
Stop-line detection
Vehicle crossing detection during RED signal
License plate extraction
OCR processing
Violation storage (plate, timestamp, image)
Annotated video generation
```
🌐 Live Deployment

You can try the live version of this project here:
🔗 https://red-light-violation-detection.onrender.com/
