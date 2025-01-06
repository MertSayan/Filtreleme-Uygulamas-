
#Muhammed Mert Sayan - 2212721028
#Nahit Furkan Öznamlı - 2212721020

from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import uuid
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Filtre uygulama fonksiyonu
def apply_filter(image, filter_type):
    if filter_type == 'thresholding':
        _, result = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return result
    elif filter_type == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        result = cv2.magnitude(sobelx, sobely)
        return cv2.convertScaleAbs(result)
    elif filter_type == 'prewitt':
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(image, -1, kernelx)
        prewitty = cv2.filter2D(image, -1, kernely)
        result = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)
        return result
    elif filter_type == 'roberts':
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = cv2.filter2D(image, -1, kernelx)
        robertsy = cv2.filter2D(image, -1, kernely)
        result = cv2.addWeighted(robertsx, 0.5, robertsy, 0.5, 0)
        return result
    elif filter_type == 'laplacian':
        result = cv2.Laplacian(image, cv2.CV_64F)
        return cv2.convertScaleAbs(result)
    elif filter_type == 'canny':
        return cv2.Canny(image, 100, 200)
    elif filter_type == 'erosion':
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.erode(image, kernel, iterations=1)
        return result
    elif filter_type == 'harris':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        result = image.copy()
        result[dst > 0.01 * dst.max()] = [0, 0, 255]
        return result
    elif filter_type == 'shi-tomasi':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        corners = np.int0(corners)
        result = image.copy()
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
        return result
    elif filter_type == 'gaussian_blur':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif filter_type == 'median_blur':
        return cv2.medianBlur(image, 5)
    elif filter_type == 'bilateral_filter':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif filter_type == 'box_blur':
        kernel = np.ones((5, 5), np.float32) / 25
        return cv2.filter2D(image, -1, kernel)
    elif filter_type == 'contour_detection':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = image.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result
    elif filter_type == 'hough_transform':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
        result = image.copy()
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a, b = np.cos(theta), np.sin(theta)
                x0, y0 = a * rho, b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return result
    elif filter_type == 'histogram_equalization':
        if len(image.shape) == 3 and image.shape[2] == 3:  # Renkli görüntü
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:  # Gri görüntü
            return cv2.equalizeHist(image)
    else:
        raise ValueError("Invalid filter type")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Content-Type'ın application/json olduğundan emin olun
        if request.is_json:
            data = request.get_json()  # JSON verisini alıyoruz
            if not data.get('image'):
                return jsonify({'error': 'No image data provided'}), 400

            # Base64 formatındaki veriyi çöz
            header, encoded = data['image'].split(',', 1)
            image_data = base64.b64decode(encoded)

            # Görüntüyü kaydet
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(image_data)

            return jsonify({'filepath': f'/static/uploads/{filename}'}), 200
        else:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/filter', methods=['POST'])
def filter_image():
    try:
        if request.is_json:
            data = request.get_json()  # JSON verisini alıyoruz
            filepath = data.get('filepath').replace('/static/uploads/', '')
            filter_type = data.get('filter')

            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
            image = cv2.imread(full_path)

            if image is None:
                return jsonify({'error': 'Invalid image file.'}), 400

            # Filtreyi uygula
            processed_image = apply_filter(image, filter_type)
            output_filename = f'filtered_{uuid.uuid4().hex}.jpg'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            cv2.imwrite(output_path, processed_image)

            return jsonify({'processed_filepath': f'/static/uploads/{output_filename}'}), 200
        else:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
