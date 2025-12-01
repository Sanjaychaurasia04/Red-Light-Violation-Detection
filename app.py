import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import re
import threading
import time
from queue import Queue
from collections import deque
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import OCR libraries
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    reader = easyocr.Reader(['en'])
except ImportError:
    EASYOCR_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'traffic_violation_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
VIOLATIONS_FOLDER = 'violations'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Create necessary directories
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, VIOLATIONS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables for processing status and violations storage
processing_status = {}
processing_queue = Queue()
violations_data = {}  # Store violations in memory {plate_number: {'count': int, 'timestamps': [], 'images': []}}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class LineDetector:
    """Line detector from the working Jupyter notebook"""
    def __init__(self, num_frames_avg=10):
        self.y_start_queue = deque(maxlen=num_frames_avg)
        self.y_end_queue = deque(maxlen=num_frames_avg)

    def detect_white_line(self, frame, color, 
                          slope1=0.03, intercept1=920, slope2=0.03, intercept2=770, 
                          slope3=-0.8, intercept3=2420):
        
        def get_color_code(color_name):
            color_codes = {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'yellow': (0, 255, 255)
            }
            return color_codes.get(color_name.lower())

        frame_org = frame.copy()
        
        # Line equations for ROI
        def line1(x): return slope1 * x + intercept1
        def line2(x): return slope2 * x + intercept2
        def line3(x): return slope3 * x + intercept3

        height, width, _ = frame.shape
        
        # Create mask for line detection
        mask1 = frame.copy()
        for x in range(width):
            y_line = line1(x)
            mask1[int(y_line):, x] = 0

        mask2 = mask1.copy()
        for x in range(width):
            y_line = line2(x)
            mask2[:int(y_line), x] = 0

        mask3 = mask2.copy()
        for y in range(height):
            x_line = line3(y)
            mask3[y, :int(x_line)] = 0

        # Convert to grayscale
        gray = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(blurred_gray)

        # Edge detection
        edges = cv2.Canny(gray, 30, 100)
        dilated_edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(dilated_edges, None, iterations=1)

        # Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=160, maxLineGap=5)

        x_start = 0
        x_end = width - 1

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + np.finfo(float).eps)
                intercept = y1 - slope * x1
                y_start = int(slope * x_start + intercept)
                y_end = int(slope * x_end + intercept)
                self.y_start_queue.append(y_start)
                self.y_end_queue.append(y_end)

        # Average y coordinates
        avg_y_start = int(sum(self.y_start_queue) / len(self.y_start_queue)) if self.y_start_queue else 0
        avg_y_end = int(sum(self.y_end_queue) / len(self.y_end_queue)) if self.y_end_queue else 0

        # Draw line
        line_start_ratio = 0.32
        x_start_adj = x_start + int(line_start_ratio * (x_end - x_start))
        avg_y_start_adj = avg_y_start + int(line_start_ratio * (avg_y_end - avg_y_start))

        mask = np.zeros_like(frame)
        cv2.line(mask, (x_start_adj, avg_y_start_adj), (x_end, avg_y_end), (255, 255, 255), 4)

        color_code = get_color_code(color)
        if color_code == (0, 255, 0):
            channel_indices = [1]
        elif color_code == (0, 0, 255):
            channel_indices = [2]
        elif color_code == (0, 255, 255):
            channel_indices = [1, 2]
        else:
            raise ValueError('Unsupported color')

        for channel_index in channel_indices:
            frame[mask[:, :, channel_index] == 255, channel_index] = 255

        # Calculate slope for mask
        slope_avg = (avg_y_end - avg_y_start) / (x_end - x_start + np.finfo(float).eps)
        intercept_avg = avg_y_start - slope_avg * x_start

        mask_line = np.copy(frame_org)
        for x in range(width):
            y_line = slope_avg * x + intercept_avg - 35
            mask_line[:int(y_line), x] = 0

        return frame, mask_line

class TrafficViolationDetector:
    def __init__(self):
        self.penalized_texts = []
        self.line_detector = LineDetector()
        
    def detect_traffic_light_color(self, image):
        """Traffic light detection from Jupyter notebook"""
        # Use the correct rectangle from notebook
        rect = (1700, 40, 100, 250)
        x, y, w, h = rect
        
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return image, 'green'
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red color detection
        red_lower = np.array([0, 120, 70])
        red_upper = np.array([10, 255, 255])
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1
        font_thickness = 2
        
        if cv2.countNonZero(red_mask) > 0:
            text_color = (0, 0, 255)
            message = "Detected Signal Status: Stop"
            color = 'red'
        elif cv2.countNonZero(yellow_mask) > 0:
            text_color = (0, 255, 255)
            message = "Detected Signal Status: Caution"
            color = 'yellow'
        else:
            text_color = (0, 255, 0)
            message = "Detected Signal Status: Go"
            color = 'green'
        
        cv2.putText(image, message, (15, 70), font, font_scale + 0.5, text_color, font_thickness + 1, cv2.LINE_AA)
        cv2.putText(image, 34 * '-', (10, 115), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        return image, color

    def extract_license_plate(self, frame, mask_line):
        """Extract license plates using the mask line approach from notebook"""
        license_plate_images = []
        
        if len(mask_line.shape) == 3:
            mask_gray = cv2.cvtColor(mask_line, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_line
        
        _, binary = cv2.threshold(mask_gray, 50, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                license_plate = frame[y:y+h, x:x+w]
                license_plate_images.append(license_plate)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return frame, license_plate_images

    def apply_ocr_to_image(self, image):
        """OCR from notebook with both Tesseract and EasyOCR"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Try Tesseract
            if TESSERACT_AVAILABLE:
                try:
                    text = pytesseract.image_to_string(binary, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    text = text.strip().replace(' ', '').replace('\n', '')
                    if len(text) >= 5:
                        formatted_text = f"{text[:2]} {text[2:]}"
                        logger.info(f"Tesseract detected: {formatted_text}")
                        return formatted_text
                except Exception as e:
                    logger.warning(f"Tesseract OCR failed: {e}")
            
            # Try EasyOCR
            if EASYOCR_AVAILABLE:
                try:
                    results = reader.readtext(binary, detail=0)
                    if results:
                        text = ' '.join(results).strip().upper()
                        text = re.sub(r'[^A-Z0-9\s]', '', text)
                        text_clean = text.replace(' ', '')
                        if len(text_clean) >= 5:
                            formatted_text = f"{text_clean[:2]} {text_clean[2:]}"
                            logger.info(f"EasyOCR detected: {formatted_text}")
                            return formatted_text
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
            
            return None
                
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return None

    def draw_penalized_text(self, frame):
        """Draw fined plates on frame"""
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 1
        font_thickness = 2
        color = (255, 255, 255)
        
        y_pos = 180
        cv2.putText(frame, 'Fined license plates:', (25, y_pos), font, font_scale, color, font_thickness)
        y_pos += 80

        for text in self.penalized_texts:
            cv2.putText(frame, '->  ' + text, (40, y_pos), font, font_scale, color, font_thickness)
            y_pos += 60

    def process_video(self, video_path, output_path, job_id):
        """Optimized video processing matching notebook logic"""
        global processing_status, violations_data
        
        vid = cv2.VideoCapture(video_path)
        
        if not vid.isOpened():
            raise Exception("Could not open video file")
        
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Process every 10th frame like the notebook (much faster!)
        frame_skip = 10
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps // frame_skip, (width, height))
        
        frame_count = 0
        self.penalized_texts = []
        violation_count = 0
        
        logger.info(f"Processing video: {width}x{height}, {fps}fps, total frames: {total_frames}")
        logger.info(f"Frame skip: {frame_skip} (processing every 10th frame)")
        
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            frame_count += 1
            
            # Skip frames like notebook
            if frame_count % frame_skip != 0:
                continue
            
            # Update progress
            progress = min(100, int((frame_count / total_frames) * 100))
            processing_status[job_id] = {
                'progress': progress,
                'violations': violation_count,
                'status': 'processing'
            }
            
            # Detect traffic light
            frame, color = self.detect_traffic_light_color(frame)
            
            # Detect line
            frame, mask_line = self.line_detector.detect_white_line(frame, color)
            
            # Process only on red light
            if color == 'red':
                frame, license_plate_images = self.extract_license_plate(frame, mask_line)
                
                for license_plate_image in license_plate_images:
                    text = self.apply_ocr_to_image(license_plate_image)
                    
                    # Check pattern and add to violations
                    if text is not None and re.match(r"^[A-Z]{2}\s[0-9]{3,4}$", text) and text not in self.penalized_texts:
                        self.penalized_texts.append(text)
                        violation_count += 1
                        logger.info(f"🚨 VIOLATION: {text}")
                        
                        # Save plate image
                        plate_filename = f"plate_{text.replace(' ', '_')}_{frame_count}.jpg"
                        plate_path = os.path.join(VIOLATIONS_FOLDER, plate_filename)
                        cv2.imwrite(plate_path, license_plate_image)
                        
                        # Store violation in memory
                        self.store_violation(text, plate_filename)
            
            # Draw penalized text
            if self.penalized_texts:
                self.draw_penalized_text(frame)
            
            out.write(frame)
            
            if frame_count % 100 == 0:
                logger.info(f"Processed frame {frame_count}/{total_frames}. Violations: {violation_count}")

        vid.release()
        out.release()
        
        processing_status[job_id] = {
            'progress': 100,
            'violations': violation_count,
            'status': 'completed',
            'result': self.penalized_texts
        }
        
        logger.info(f"✅ Complete: {violation_count} violations")
        return self.penalized_texts

    def store_violation(self, plate_number, image_filename):
        """Store violation data in memory"""
        global violations_data
        
        if plate_number not in violations_data:
            violations_data[plate_number] = {
                'count': 0,
                'timestamps': [],
                'images': []
            }
        
        violations_data[plate_number]['count'] += 1
        violations_data[plate_number]['timestamps'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        violations_data[plate_number]['images'].append(image_filename)

def get_violations():
    """Get all violations from memory"""
    # Convert to list format sorted by count
    result = [(plate, data['count']) for plate, data in violations_data.items()]
    result.sort(key=lambda x: x[1], reverse=True)
    return result

def get_violation_details(plate_number):
    """Get detailed violation information for a specific plate"""
    return violations_data.get(plate_number, None)

def process_queue():
    """Background thread to process videos"""
    while True:
        if not processing_queue.empty():
            job_id, input_path, output_path = processing_queue.get()
            try:
                logger.info(f"Starting job {job_id}")
                detector = TrafficViolationDetector()
                detector.process_video(input_path, output_path, job_id)
                logger.info(f"Completed job {job_id}")
            except Exception as e:
                processing_status[job_id] = {
                    'progress': 0,
                    'violations': 0,
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"Error in job {job_id}: {e}")
        time.sleep(1)

# Start processing thread
processing_thread = threading.Thread(target=process_queue, daemon=True)
processing_thread.start()

# Flask routes
@app.route('/')
def index():
    violations = get_violations()
    return render_template('index.html', 
                         violations=violations,
                         TESSERACT_AVAILABLE=TESSERACT_AVAILABLE,
                         EASYOCR_AVAILABLE=EASYOCR_AVAILABLE)

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['video']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            output_filename = 'processed_' + filename
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            
            file.save(input_path)
            
            job_id = str(int(time.time()))
            processing_queue.put((job_id, input_path, output_path))
            
            processing_status[job_id] = {
                'progress': 0,
                'violations': 0,
                'status': 'queued',
                'filename': output_filename
            }
            
            flash('Video uploaded! Processing started.')
            return redirect(url_for('processing_status_page', job_id=job_id))
        else:
            flash('Invalid file type')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/processing/<job_id>')
def processing_status_page(job_id):
    return render_template('processing.html', job_id=job_id)

@app.route('/api/status/<job_id>')
def get_processing_status(job_id):
    status = processing_status.get(job_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/results/<job_id>')
def show_results(job_id):
    status = processing_status.get(job_id, {})
    if status.get('status') == 'completed':
        return render_template('results.html',
                            plates=status.get('result', []),
                            video_filename=status.get('filename', ''),
                            total_violations=len(status.get('result', [])))
    elif status.get('status') == 'error':
        flash(f'Error: {status.get("error", "Unknown")}')
        return redirect('/')
    else:
        flash('Processing not complete')
        return redirect(url_for('processing_status_page', job_id=job_id))

@app.route('/demo')
def demo():
    demo_video_path = 'traffic_video.mp4'
    if not os.path.exists(demo_video_path):
        flash('Demo video not found')
        return redirect('/')
    
    output_filename = 'processed_demo.mp4'
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    job_id = f"demo_{int(time.time())}"
    processing_queue.put((job_id, demo_video_path, output_path))
    
    processing_status[job_id] = {
        'progress': 0,
        'violations': 0,
        'status': 'queued',
        'filename': output_filename
    }
    
    flash('Demo processing started!')
    return redirect(url_for('processing_status_page', job_id=job_id))

@app.route('/download/<filename>')
def download_video(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('File not found')
        return redirect('/')

@app.route('/violations')
def view_violations():
    violations = get_violations()
    violations_detail = []
    for plate, count in violations:
        detail = get_violation_details(plate)
        violations_detail.append({
            'plate': plate,
            'count': count,
            'timestamps': detail['timestamps'],
            'images': detail['images']
        })
    return render_template('violations.html', violations=violations_detail)

@app.route('/violation_image/<filename>')
def violation_image(filename):
    return send_file(os.path.join(VIOLATIONS_FOLDER, filename))

@app.route('/clear_data', methods=['POST'])
def clear_data():
    """Clear all violation data"""
    global violations_data
    
    try:
        # Clear in-memory data
        violations_data = {}
        
        # Clear violation images
        for f in os.listdir(VIOLATIONS_FOLDER):
            if f.endswith('.jpg'):
                os.remove(os.path.join(VIOLATIONS_FOLDER, f))
                
        flash('All violation data cleared!')
    except Exception as e:
        flash(f'Error: {str(e)}')
    
    return redirect('/')

@app.route('/api/violations')
def api_violations():
    """API endpoint to get violations as JSON"""
    violations = get_violations()
    violations_detail = []
    for plate, count in violations:
        detail = get_violation_details(plate)
        violations_detail.append({
            'plate_number': plate,
            'violation_count': count,
            'timestamps': detail['timestamps'],
            'images': detail['images']
        })
    return jsonify(violations_detail)

if __name__ == '__main__':
    logger.info("🚀 Traffic Violation Detection System Started!")
    logger.info("🌐 http://localhost:5000")
    logger.info("📝 All data stored in memory (no database)")
    app.run(debug=True, host='0.0.0.0', port=5000)