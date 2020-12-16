""" Demo of SSD detection with MOSSE tracking
Performs detection (at a specified frequency) and tracking on a given video.

usage: demo_ssd_mosse.py [INPUT] [TRAINED_MODEL] [-dt DET_THRESHOLD] [-o OUTPUT] [-d DISPLAY]

required arguments:
    INPUT
        Path to the input video, must either a directory containing jpg files, or an .mp4 file.
    
    TRAINED_MODEL
        Path to the pipeline.config file used for training the detection model.

    DS_MODEL
        Path to the DeepSORT model file.

optional arguments:
    -dt DET_THRESHOLD
            Minimum required score for detections to be considered.

    -o OUTPUT
            Path to output directory. If a path is specified, results will be saved there.
    
    -d DISPLAY
            Specify if results must be displayed in real time.
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
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to the input video, must either a directory containing jpg files, or an .mp4 file.")
parser.add_argument("trained_model", type=str, help="Path to the pipeline.config file used for training the detection model.")
parser.add_argument("ds_model", type=str, help="Path to the DeepSORT model file.")
parser.add_argument("-dt", "--det_thld", type=float, default=0.4, help="Minimum required score for detections to be considered.")
parser.add_argument("-o", "--output", type=str, help="Path to output directory.")
parser.add_argument("-d", "--display", action='store_true', help="Specify if results must be displayed in real time.")
args = parser.parse_args()

# Build a detection model from the pipeline config, and restore last checkpoint
configs = config_util.get_configs_from_pipeline_file(os.path.join(args.trained_model, "pipeline.config"))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt_path = args.trained_model
ckpt.restore(tf.train.latest_checkpoint(ckpt_path)).expect_partial()

@tf.function
def detect(image_tensor):
    image, shapes = detection_model.preprocess(image_tensor)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def run():

    # Initialize the DeepSORT tracker
    max_cosine_distance = 0.6
    nn_budget = None
    model_filename = args.ds_model
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

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
        
        # Detection
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        
        mask = detections['detection_scores'] >= args.det_thld
        scores = ["{:.2f}".format(score) for score in detections['detection_scores'][mask]]
        boxes = detections['detection_boxes'][mask]
        
        tlwh_boxes = []
        for j in range(len(boxes)):
            ymin,xmin,ymax,xmax = boxes[j]
            xmin = int(round(xmin * width))
            ymin = int(round(ymin * height))
            xmax = int(round(xmax * width))
            ymax = int(round(ymax * height))
            w = (xmax - xmin)
            h = (ymax - ymin)
            tlwh_boxes.append([xmin, ymin, w, h])

        # Tracking
        features = np.array(encoder(frame, tlwh_boxes))
        detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(tlwh_boxes, scores, features)]
        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks and draw then on the frame
        for track in tracker.tracks:
            if (not track.is_confirmed()) or (track.time_since_update > 3):
                continue
            # Get the corrected/predicted bounding box.
            bbox = track.to_tlwh()
            # Get the ID for the particular track.
            tracking_id = track.track_id
            x,y,w,h = [int(v) for v in bbox.tolist()]
            cv2.rectangle(image_np, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(image_np, "Person " + str(tracking_id), (x, y-5), FONT, font_size, green, thickness)

        # Time for performing detection and write the results on the frame.
        FPS = round(1.0 / (time.time() - frame_start_time), 2)
        fps_arr.append(FPS)
        cv2.putText(image_np, str(FPS) + " FPS" , (5, 40), FONT, 3, green, 2)
        image_name = "img"+str(i)+".jpg"

        if(args.display):
            cv2.imshow("SSD and DeepSORT", cv2.resize(image_np, (disp_width, disp_height)))
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