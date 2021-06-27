"""

"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import concurrent.futures




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inputdir", required=True, type=str,
                help="path to input directory; folder structure should be <subfolders>/<class_name>.[mp4,mov...]")
ap.add_argument("-o", "--outputdir", required=False,
                default='data',
                help="path to save outputfiles")
ap.add_argument("-n", "--nframes", required=False, type=int,
                default='-1',
                help="enter no of frames to consider for landmark \n"+
                     " extraction in each video file, DEFAULT it will process whole video",
                )
ap.add_argument("-f", "--format", required=False, type=str,
                choices=["csv", "npy"],
                default='csv',
                help="file format to save landmarks of video frames, DEFAULT 'csv'"
                )
ap.add_argument("-p", "--nprocesses", required=False, type=int,
                help="provide number of parallel processes for processing videos"
                )

args = vars(ap.parse_args())


mp_face_mesh = mp.solutions.face_mesh

def generateFacemeshnumpyArray(target_dir, path,no,file_type):
    '''
    'no' number of landmarks are saved as 'file_type' files
    :param target_dir: output directory directory
    :param path: video file path
    :param no: no of frames to convert landmarks for, -1 to convert full video
    :param file_type: [csv or npy]
    :return:
    '''
    cap = cv2.VideoCapture(path)

    print('total frames=',cap.get(cv2.CAP_PROP_FRAME_COUNT))

    i=0
    last_i =-100
    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

      #check if video stream is opened and no of frames processed are less than 'no'
      # or process all frames if -1 is passes as 'no'
      while cap.isOpened() and (i < no or no<=0):
        i += 1
        target =None

        #check if frame's landmarks is already saved at the target location and skip this frame
        target = target_dir + '_' + str(i)
        if os.path.exists(target+'.'+file_type):
            continue
        if last_i +10 < i:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # setting frame position from where to resume reading video
        last_i = i
        print('writing landmarks for=',target)

        #reading image from video at given set frame
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # breaking loop if end of video frames are reached
            break


        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
          for face_landmarks in results.multi_face_landmarks:
            landmark_arr = []
            #yeilding landmark array and 'target' ie folder location to save
            for point in face_landmarks.landmark:
                landmark_arr.append(np.array([point.x, point.y, point.z]))
            save_landmark_csv( landmark_arr, target, file_type)

    cap.release()
    return 'completed ='+path

def save_landmark_csv(ar, path,file_type):
    '''
    saves ar at given path with file type = file_type
    :param ar: landmarks array
    :param path: file path without extension
    :param file_type: csv or npy
    :return:
    '''
    if file_type == 'npy':
        np.save(path,ar)
    else:
        # csv file has three columns x, y,z and 468 lines for each landmark point
        with open(path+"."+file_type,'w') as f:
         for point in ar:
             point = point*1000
             f.write(str(int(point[0]))+","+str(int(point[1]))+","+str(int(point[2]))+"\n")

def get_video_target_paths(rootdir):
    '''
    :param rootdir:
    :return: returns dictionary of (video path, target location without index and file type)
    '''
    # iterate through each video file in subdirectory
    video_target ={}
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(subdir, file)
            print(path)
            label = os.path.basename(path)[0]  # extracting label name from file name
            basedir = os.path.basename(subdir)  # extracting subfolder name
            target_dir = os.path.join(outputdir, label)  # target folder has labels as subdirectories
            if not os.path.exists(target_dir):  # create target_dir if not exixts
                os.makedirs(target_dir)
            target = os.path.join(target_dir, basedir)  # output file path without extension and index
            video_target[path] = target

    return video_target


def extract_landmarks(rootdir, outputdir, no_frames, file_type, no_processes):
    '''
    iterate through video files in given root_dir and
    saves landmarks files in outputdir with subfolders for each class.

    Video files should be in format 'rootdir'/<subfolder>/<class_name>.<[mp4,mov...]>

    It has the capability to resume from the video and frame where landmark generation was stopped
    so it will save time by not overriding landmark files and reading all processed frames from videos
    :param rootdir: folder which conatins videos with above folder structure
    :param outputdir: path to save landmark files
    :param no_frames: int to convert frames till this number
    :param file_type: ['csv' or 'npy']
    '''
    #create output directory if not exixts
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    status = True

    video_target = get_video_target_paths(rootdir)
    with concurrent.futures.ProcessPoolExecutor(no_processes) as executor:
        futures = []
        future_to_path = {executor.submit(generateFacemeshnumpyArray, target_dir = target, path = path, no = no_frames,
                            file_type=file_type): target for  path, target in video_target.items()}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            print(future.done())
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (path, exc))
                status = False
            else:
                print('%r page is %d bytes' % (path, len(data)))

    return  status

#input data directory
rootdir = args['inputdir']
file_type = args['format']
outputdir = args['outputdir']
no_frames = args['nframes']
no_processes = args['nprocesses']

if __name__=="__main__":
    import time

    threaded_start = time.time()

    while(not extract_landmarks(rootdir,outputdir,no_frames,file_type,no_processes)):
        pass

    #printing few samples from each class folder
    for subdir, dirs, files in os.walk(outputdir):
        for file in files:
            file = os.path.join(subdir,file)
            if file_type == 'npy':
                ar = np.load(file,allow_pickle=True)
                print(ar[0])
            else:
                with open(file) as f:
                    print(f.readline())
            break

    print("Parallel processing time:", time.time() - threaded_start)
