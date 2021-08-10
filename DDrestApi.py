from flask import Flask, render_template, request,jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import imutils
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
# run_with_ngrok(app)

mp_face_mesh = mp.solutions.face_mesh
model_path = r'output\DD2CNN_model.h5'
model = tf.keras.models.load_model(model_path)

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
      f = request.files['file'].read()
      npimg = np.frombuffer(f, np.uint8)
      img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

      frame = imutils.resize(img, width=450)
      image = frame
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      status = None
      try:
          with mp_face_mesh.FaceMesh() as face_mesh:
            results = face_mesh.process(image)
            for face_landmarks in results.multi_face_landmarks:
                landmark_arr = []
                for point in face_landmarks.landmark:
                    landmark_arr.append([point.x, point.y, point.z])
                # print(landmark_arr)
                status = model.predict([landmark_arr])
      except:
          pass
      print(status)
      msg = 'alert'
      if status is not None:
          index = np.argmax(status[0])
          percentage = status[0][index]
          label = ['alert', 'drowsy', 'semi alert']
          msg = label[index] + ' with confidence ' + '{:.2f}'.format(percentage)
      cv2.imwrite('image.png',img)
      return jsonify({'result': msg })
  else:
      return ''
if __name__ == '__main__':
   app.run()