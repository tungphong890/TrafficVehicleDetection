import os
import uuid
import shutil
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np

# YOLOv8 import
from ultralytics import YOLO

# Flask config
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

# Class names
CLASS_NAMES = ['__background__','Car','Number Plate','Blur Number Plate','Two Wheeler','Auto','Bus','Truck']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.secret_key = 'secret!'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load models at startup (CPU only)
MODELS = {}

def load_yolov8_model():
    model = YOLO('yolov8.pt')
    model.fuse()
    return model

def load_fasterrcnn_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES))
    checkpoint = torch.load('faster_rcnn.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def load_ssd_model():
    import torchvision
    model = torchvision.models.detection.ssd300_vgg16(weights=None, num_classes=8)
    checkpoint = torch.load('ssd.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    return model

MODELS['yolov8'] = load_yolov8_model()
MODELS['fasterrcnn'] = load_fasterrcnn_model()
MODELS['ssd'] = load_ssd_model()

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def random_filename(filename):
    ext = filename.rsplit('.', 1)[1]
    return f"{uuid.uuid4().hex}.{ext}"

def cleanup_folder(folder):
    for f in os.listdir(folder):
        file_path = os.path.join(folder, f)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception:
            pass

def draw_boxes_pil(image, boxes, labels, scores, class_names, score_threshold=0.25):
    """Draw bounding boxes using PIL, returns PIL Image."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box
        color = (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{class_names[label]}: {score:.2f}" if label < len(class_names) else str(label)
        draw.text((x1, y1-10 if y1-10>0 else y1+5), label_text, fill=color, font=font)
    return image

def process_image_with_yolov8(image_path, result_path):
    model = MODELS['yolov8']
    # Ultralytics YOLO expects numpy image or path
    results = model.predict(source=image_path, imgsz=640, conf=0.25, device='cpu')
    result = results[0]
    img = cv2.imread(image_path)
    for box, label, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
        if score < 0.25:
            continue
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        class_name = CLASS_NAMES[label+1] if label+1 < len(CLASS_NAMES) else str(label)
        text = f"{class_name}: {score:.2f}"
        cv2.putText(img, text, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(result_path, img)
    return

def process_image_with_fasterrcnn(image_path, result_path):
    model = MODELS['fasterrcnn']
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    pil_img = image.copy()
    pil_img = draw_boxes_pil(pil_img, boxes, labels, scores, CLASS_NAMES, score_threshold=0.25)
    pil_img.save(result_path)
    return

def process_image_with_ssd(image_path, result_path):
    model = MODELS['ssd']
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    pil_img = image.copy()
    pil_img = draw_boxes_pil(pil_img, boxes, labels, scores, CLASS_NAMES, score_threshold=0.25)
    pil_img.save(result_path)
    return

def extract_video_preview(input_video, output_gif, model_key, frame_sample_rate=8, max_frames=16):
    """
    Process video, draw predictions on sampled frames, save as GIF.
    """
    cap = cv2.VideoCapture(input_video)
    frames = []
    count = 0
    sampled = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while sampled < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_sample_rate == 0:
            # Resize for inference
            img = cv2.resize(frame, (640, 640))
            temp_path = f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(temp_path, img)
            result_path = f"{uuid.uuid4().hex}_res.jpg"
            if model_key == 'yolov8':
                process_image_with_yolov8(temp_path, result_path)
            elif model_key == 'fasterrcnn':
                process_image_with_fasterrcnn(temp_path, result_path)
            elif model_key == 'ssd':
                process_image_with_ssd(temp_path, result_path)
            img_res = cv2.imread(result_path)
            frames.append(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
            os.remove(temp_path)
            os.remove(result_path)
            sampled += 1
        count += 1
        if count > total_frames:
            break
    cap.release()
    # Make GIF using PIL
    pil_frames = [Image.fromarray(f) for f in frames]
    if pil_frames:
        pil_frames[0].save(output_gif, save_all=True, append_images=pil_frames[1:], duration=200, loop=0)
    return

@app.route('/', methods=['GET', 'POST'])
def index():
    cleanup_folder(app.config['UPLOAD_FOLDER'])
    cleanup_folder(app.config['RESULT_FOLDER'])
    result_url = None
    preview_type = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        model_key = request.form.get('model')
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename) and model_key in MODELS:
            filename = random_filename(secure_filename(file.filename))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            ext = filename.rsplit('.', 1)[1].lower()
            result_fn = f"result_{uuid.uuid4().hex}"
            if ext in {'jpg', 'jpeg', 'png'}:
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_fn + '.jpg')
                # Resize for model input
                img = Image.open(file_path).convert("RGB")
                img = img.resize((640, 640))
                img.save(file_path)
                if model_key == 'yolov8':
                    process_image_with_yolov8(file_path, result_path)
                elif model_key == 'fasterrcnn':
                    process_image_with_fasterrcnn(file_path, result_path)
                elif model_key == 'ssd':
                    process_image_with_ssd(file_path, result_path)
                result_url = url_for('static', filename=f"results/{os.path.basename(result_path)}")
                preview_type = 'image'
            elif ext == 'mp4':
                result_path = os.path.join(app.config['RESULT_FOLDER'], result_fn + '.gif')
                extract_video_preview(file_path, result_path, model_key)
                result_url = url_for('static', filename=f"results/{os.path.basename(result_path)}")
                preview_type = 'gif'
            else:
                flash('File type not supported')
                return redirect(request.url)
            # Clean up upload after serving
            try:
                os.remove(file_path)
            except Exception:
                pass
            return render_template('index.html', result_url=result_url, preview_type=preview_type)
        else:
            flash('Invalid file or model selection')
            return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)