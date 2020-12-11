""" Demo of SSD detection with MOSSE tracking
Performs detection (at a specified frequency) and tracking on a given video.

usage: demo_ssd_mosse.py [INPUT] [CONFIG] [CKPT] [DET_FREQ] [DET_THRESHOLD] [OUTPUT] [LIVE_DISPLAYING]

optional arguments:
    -f DET_FREQ
            Frequency at which to perform detections (for example 10 to perform detection every 10 frames).

    -dt DET_THRESHOLD
            Minimum required score for detections to be considered.

    -o OUTPUT
            Path to output directory. If a path is specified, results will be saved there.
    
    -ld LIVE_DISPLAYING
            Specify if results must be displayed in real time.

required arguments:
    INPUT
        Path to the input video, must either a directory containing jpg files, or an .mp4 file.
    
    CONFIG
        Path to the pipeline.config file used for training the detection model.

    CKPT
        Path to directory containing training checkpoints.

"""

import os
from os.path import isdir, isfile, join
import argparse
import cv2
import numpy as np
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
#Google Colab does not support cv2.imshow and the following method must be used instead
#from google.colab.patches import cv2_imshow


parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to the input video, must either a directory containing jpg files, or an .mp4 file.")
parser.add_argument("config", type=str, help="Path to the pipeline.config file used for training the detection model.")
parser.add_argument("ckpt", type=str, help="Path to directory containing training checkpoints.")
parser.add_argument("-f", "--det_freq", type=int, default=10, help="Frequency at which to perform detections.")
parser.add_argument("-dt", "--det_thld", type=float, default=0.4, help="Minimum required score for detections to be considered.")
parser.add_argument("-o", "--output", type=str, help="Path to output directory.")
parser.add_argument("-d", "--display", action='store_true', help="Specify if results must be displayed in real time.")
args = parser.parse_args()

# Build a detection model from the pipeline config, and restore last checkpoint
configs = config_util.get_configs_from_pipeline_file(args.config)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(tf.train.latest_checkpoint(args.ckpt))).expect_partial()

@tf.function
def detect(image_tensor):
    image, shapes = detection_model.preprocess(image_tensor)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def run():

    # Parameters for info displaying
    FONT = cv2.FONT_HERSHEY_PLAIN
    green = (0, 255, 0)
    thickness = 1
    font_size = 1.5

    # Initialize capture
    if isdir(args.input):
        cap = cv2.VideoCapture(join(args.input, "%06d.jpg"))
    else :
        cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    disp_width = round(0.5*width)
    disp_height = round(0.5*height)

    # initialize OpenCV's special multi-object tracker
    trackers = cv2.MultiTracker_create()

    boxes = []
    scores = []

    fps_arr = []
    i = 0
    
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        
        frame_start_time = time.time()
        
        image_np = np.array(frame)

        if(i % 10 == 0):
            
            image_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect(image_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections
            
            trackers = cv2.MultiTracker_create()
            mask = detections['detection_scores'] >= args.det_thld
            scores = ["{:.2f}".format(score) for score in detections['detection_scores'][mask]]
            boxes = detections['detection_boxes'][mask]
            
            for j in range(len(boxes)):
                ymin,xmin,ymax,xmax = boxes[j]
                xmin = int(round(xmin * width))
                ymin = int(round(ymin * height))
                xmax = int(round(xmax * width))
                ymax = int(round(ymax * height))
                w = (xmax - xmin)
                h = (ymax - ymin)
                
                # set a tracker for this box
                mosse_tracker = cv2.TrackerMOSSE_create()
                trackers.add(mosse_tracker, frame, (xmin, ymin, w, h))
                
                # draw the bounding box and its label on the frame
                cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(image_np, "person " + str(scores[j]), (xmin, ymin-5), FONT, font_size, green, thickness)

        else :   
            # grab the updated bounding box coordinates (if any) for each
            # object that is being tracked
            _, boxes = trackers.update(frame)

            # loop over the bounding boxes and draw then on the frame
            for j in range(len(boxes)):
                x,y,w,h = [int(v) for v in boxes[j]]
                cv2.rectangle(image_np, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(image_np, "person " + str(scores[j]), (x, y-5), FONT, font_size, green, thickness)
        
        FPS = round(1.0 / (time.time() - frame_start_time), 2)
        fps_arr.append(FPS)
        cv2.putText(image_np, str(FPS) + " FPS" , (5, 40), FONT, 3, green, 2)
        image_name = "img"+str(i)+".jpg"

        if(args.display):
            cv2.imshow("SSD and MOSSE", cv2.resize(image_np, (disp_width, disp_height)))
            #cv2_imshow(cv2.resize(image_np, (disp_width, disp_height)))
            key = cv2.waitKey(1)
            if key==27:
                break
        if(not args.output is None):
            cv2.imwrite(join(args.output, image_name), image_np)
        
        i += 1

    AVG_FPS = np.mean(fps_arr)
    print("Average FPS :", AVG_FPS)
    cap.release()
    cv2.destroyAllWindows()
    print("Finished")

if __name__ == '__main__':
    run()
