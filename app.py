from flask import Flask, render_template, Response
from utils.product_detection import detect_products
from utils.ocr_scanning import scan_text
from utils.freshness_detection import detect_freshness

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/product-detection')
def product_detection():
    return render_template('product_detection.html')

@app.route('/ocr-scanning')
def ocr_scanning():
    return render_template('ocr_scanning.html')

@app.route('/freshness-detection')
def freshness_detection():
    return render_template('freshness_detection.html')

@app.route('/video_feed/<mode>')
def video_feed(mode):
    if mode == 'product':
        return Response(detect_products(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif mode == 'ocr':
        return Response(scan_text(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif mode == 'freshness':
        return Response(detect_freshness(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
