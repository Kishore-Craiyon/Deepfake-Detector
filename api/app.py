from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from pathlib import Path
import logging
from inference import DeepFakeDetector
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize detector
MODEL_PATH = 'models/best_model.pth'
CONFIG_PATH = 'config/config.yaml'

try:
    detector = DeepFakeDetector(MODEL_PATH, CONFIG_PATH)
    logger.info("DeepFake detector initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize detector: {e}")
    detector = None

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None
    })

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    """Main deepfake detection endpoint"""
    if detector is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Check if file was uploaded
        if 'video' not in request.files:
            return jsonify({
                'error': 'No video file provided'
            }), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'File type not allowed. Supported: mp4, avi, mov, mkv'
            }), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, 
                                       suffix='.mp4') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Run detection
            result = detector.predict(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return jsonify({
                'success': True,
                'result': result
            })
            
        except Exception as e:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error in detection: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Detection failed: {str(e)}'
        }), 500

@app.route('/batch_detect', methods=['POST'])
def batch_detect():
    """Batch detection endpoint"""
    if detector is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    try:
        files = request.files.getlist('videos')
        
        if not files:
            return jsonify({
                'error': 'No video files provided'
            }), 400
        
        results = []
        
        for file in files:
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'error': 'File type not allowed'
                })
                continue
            
            # Save temporarily and process
            with tempfile.NamedTemporaryFile(delete=False, 
                                           suffix='.mp4') as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                result = detector.predict(temp_path)
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in batch detection: {str(e)}")
        return jsonify({
            'error': f'Batch detection failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
