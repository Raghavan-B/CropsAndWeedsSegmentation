from flask import Flask, render_template, request, jsonify, flash
import os
from PIL import Image
from werkzeug.utils import secure_filename
from src.cropsAndWeedsSegmentation.pipeline.prediction_pipeline import PredictionPipeline
from pathlib import Path
from io import BytesIO
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg','jpeg'}

os.makedirs(UPLOAD_FOLDER,exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 *1024*1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def segment_weeds(image_path,filename):
    obj = PredictionPipeline(model_weights_path=Path('artifacts/model_trainer/model_weights.pth'))
    pred_mask = obj.segment_images(image_path)
    pred_mask = Image.fromarray(pred_mask)
    print('Prediction done!!')

    buffer_segmented = BytesIO()
    pred_mask.save(buffer_segmented,format='PNG')
    buffer_segmented.seek(0)
    segmented_b64 = base64.b64encode(buffer_segmented.read()).decode('utf-8')
    return segmented_b64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/segment',methods=['GET','POST'])
def segment():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({'error':'No file uploaded'}),400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error':'No selected file'}),400
        
        if not(file) or not allowed_file(file.filename):
            return jsonify({'error':'Only JPEG/JPG images are allowed'}),400
        
        print('file has been collected successfully!!')
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)
        file.save(file_path)
        print('file saved successfully')
        buffer_segmented = BytesIO()
        img = Image.open(file_path)
        img.save(buffer_segmented,format='PNG')
        buffer_segmented.seek(0)
        image_b64 = base64.b64encode(buffer_segmented.read()).decode('utf-8')

        segmented_b64 = segment_weeds(image_path=file_path,filename = filename)
        os.remove(file_path)

        return jsonify({
            'original_image':f'data:image/png;base64,{image_b64}',
            'segmented_image': f'data:image/png;base64,{segmented_b64}'
        })
    else:
        return render_template('segment.html')

if __name__ == '__main__':
    app.run(debug=True)
