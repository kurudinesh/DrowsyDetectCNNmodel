import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
import numpy as np
import cv2
import platform
import tensorflow as tf
from ineuron import  config
import os

# For static images:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5) as face_mesh:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print and draw face mesh landmarks on the image.
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()
    for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACE_CONNECTIONS,
          landmark_drawing_spec=drawing_spec,
          connection_drawing_spec=drawing_spec)
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)





if __name__ == "__main__":
    model_path = r'output\DD2CNN_model.h5'
    model = tf.keras.models.load_model(model_path)

    print("python version:", platform.python_version())
    cap = cv2.VideoCapture(0)
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh() as face_mesh:
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            image = frame
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            status = None
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACE_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec)
                    landmark_arr = []
                    for point in face_landmarks.landmark:
                        landmark_arr.append([point.x, point.y, point.z])
                    print(landmark_arr)
                    status = model.predict([landmark_arr])
            print(status)
            index = np.argmax(status[0])
            percentage = status[0][index]
            frame = image
            label = ['alert','drowsy','semi alert']
            msg = label[index]+' '+'{:.2f}'.format(percentage)
            cv2.putText(frame,msg,(400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

            # Process the keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("quit")
                break
            # show the images
            cv2.imshow('frame',frame)

    cap.release()
    cv2.destroyAllWindows()