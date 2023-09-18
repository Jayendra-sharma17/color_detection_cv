from flask import Flask, render_template, Response
import cv2
import skimage as ski
import numpy as np

app = Flask(__name__)

def generate_frames():
    vid = cv2.VideoCapture(0)  # Open the camera (0 represents default camera)
    while True:
        ack ,img = vid.read()
        if not ack:
            break
        else:
            th,red_bw=cv2.threshold(cv2.subtract(
                img[:,:,-1],cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ),80,255,cv2.THRESH_BINARY)

            strel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            red_bw=cv2.morphologyEx(red_bw,cv2.MORPH_CLOSE,strel,1)

            red_bw=ski.morphology.remove_small_objects(red_bw.astype(bool) , 3000) .astype('uint8')*255

            red_bw=ski.morphology.remove_small_holes(red_bw.astype(bool) , 3000) .astype('uint8')*255

    
            rps=ski.measure.regionprops(ski.measure.label(red_bw.astype('uint8')))

            count=len(rps)

            img_copy=img.copy()

            cv2.putText(img_copy,str(count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,0),5)

            for rp in rps:
                y1,x1,y2,x2=rp.bbox

                cv2.rectangle(img_copy,(x1,y1),(x2,y2),(0,0,255),5)

            ack, buffer = cv2.imencode('.jpg', img_copy)
            img_copy = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_copy + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
