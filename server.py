import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from darkflow.net.build import TFNet
import cv2

UPLOAD_FOLDER = '/home/rohitner'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
options = {"model": "/home/rohitner/tensorflow/darkflow/cfg/tiny-yolo-voc.cfg", "load": "/home/rohitner/tensorflow/darkflow/bin/tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 0.5}
tfnet = TFNet(options)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            imgcv = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            results = tfnet.return_predict(imgcv)
            for result in results:
                cv2.rectangle(imgcv,
                               (result["topleft"]["x"], result["topleft"]["y"]),
                               (result["bottomright"]["x"], result["bottomright"]["y"]),
                             (255, 0, 0), 4)
                text_x, text_y = result["topleft"]["x"] - 10, result["topleft"]["y"] - 10

                cv2.putText(imgcv, result["label"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename), imgcv)
            return redirect(url_for('send_file', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)

