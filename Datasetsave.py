import tensorflow as tf


"""

"""

import cv2
import mediapipe as mp
import numpy as np
import os
import argparse
import concurrent.futures
import tensorflow as tf
import glob
import json


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
                choices=["csv", "npy",'dataframe'],
                default='dataframe',
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

    flag_continue = True
    i = 0
    last_i = -100

    while flag_continue:
        try:

            cap = cv2.VideoCapture(path)

            print(path,' has total frames=',cap.get(cv2.CAP_PROP_FRAME_COUNT),'setting frame no to',i)

            json_file = os.path.join(target_dir,'landmarks.json')

            if os.path.exists(json_file):
                print('checking no of lined present')
                with open(json_file) as f:
                    for _ in f.readlines():
                        i=i+1
            else:
                os.makedirs(target_dir)

            if last_i + 10 < i:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # setting frame position from where to resume reading video
            last_i = i


            with mp_face_mesh.FaceMesh(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
              with open(json_file,'a+') as f:
                  #check if video stream is opened and no of frames processed are less than 'no'
                  # or process all frames if -1 is passes as 'no'
                  while cap.isOpened() and (i < no or no<=0):
                    i += 1
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
                            landmark_arr.append([point.x, point.y, point.z])
                        f.write(json.dumps(landmark_arr)+"\n")
        except Exception as e:
            print('Error in frame', i, e)
        except:
            print('Error in frame', i)
        else:
            print("Completed landmark extraction for file",path)
            flag_continue = False


    save_landmark_csv(target_dir,file_type)
    cap.release()
    return 'completed ='+path

def save_landmark_csv(path,file_type):
    '''
    saves ar at given path with file type = file_type
    :param ar: landmarks array
    :param path: file path without extension
    :param file_type: csv or npy
    :return:
    '''

    def custom_shard(element):
        return tf.constant(0,dtype=tf.int64)

    json_file = os.path.join(path, 'landmarks.json')

    ar =[]
    with open(json_file) as f:
        for line in f.readlines():
            ar.append(json.loads(line))

    ar = np.float16(ar)
    print(ar.shape,ar.dtype)
    ds = tf.data.Dataset.from_tensor_slices(ar)
    for item in ds.take(1):
        print(item)
    tf.data.experimental.save(ds,path,'GZIP',custom_shard)
    os.remove(json_file)
    # tf.data.experimental.save(ds, path)

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
        future_to_path = {}
        for path, target in list(video_target.items()):
            sp = os.path.join(target, '*', "*", "*.snapshot")
            if glob.glob(sp):
                continue
            print(path,target)
            # generateFacemeshnumpyArray(target, path, no_frames, file_type)
            future = executor.submit(generateFacemeshnumpyArray, target_dir = target, path = path, no = no_frames,
                            file_type=file_type)
            future_to_path[future] = target
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
ds = []

if __name__=="__main__":
    import time

    threaded_start = time.time()

    while(not extract_landmarks(rootdir,outputdir,no_frames,file_type,no_processes)):
        pass

    # path = r'C:\Users\kurud\Documents\ineaurondeeplearn\internship\DrowsyDetectCNNmodel\data3\5\**'
    # sp = os.path.join(path, '*', "*", "*.snapshot")
    # print(sp)
    #
    #
    # if glob.glob(sp):
    #     print("snapshot exists")
    #
    #     typespec = tf.TensorSpec(
    #         shape=[468,3], dtype=tf.dtypes.double, name=None
    #     )
    #     ds = tf.data.experimental.load(path,compression='GZIP')
    #     tf.print("size=",tf.data.experimental.cardinality(ds))
    #     for item in ds.take(1).as_numpy_iterator():
    #         tf.print("printing dataset")
    #         tf.print(item)
    print("Parallel processing time:", time.time() - threaded_start)



