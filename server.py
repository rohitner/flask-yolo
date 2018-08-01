from flask import Flask, url_for, render_template, Response
from darkflow.net.build import TFNet
import cv2

app = Flask(__name__)
options = {"model": "/home/rohitner/tensorflow/darkflow/cfg/tiny-yolo-voc.cfg",
           "load": "/home/rohitner/tensorflow/darkflow/bin/tiny-yolo-voc.weights",
           "threshold": 0.1, "gpu": 0.8}
tfnet = TFNet(options)

def gen(camera):
    while True:
        success, imgcv = camera.read()
        results = tfnet.return_predict(imgcv)
        for result in results:
            cv2.rectangle(imgcv,
                           (result["topleft"]["x"], result["topleft"]["y"]),
                           (result["bottomright"]["x"], result["bottomright"]["y"]),
                         (255, 0, 0), 4)
            text_x, text_y = result["topleft"]["x"] - 10, result["topleft"]["y"] - 10

            cv2.putText(imgcv, result["label"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
        ret, jpeg = cv2.imencode('.jpg', imgcv)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    cam = cv2.VideoCapture(0)
    return Response(gen(cam),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def webcam():
    return render_template('webcam.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)

