""" Demo of YOLOv3 detection with KCF tracking
Performs detection (at a specified frequency) and tracking on a given video.

usage: demo_yolo_kcf.py [INPUT] [-o OUTPUT] [-d DISPLAY]

required arguments:
    INPUT
        Path to the input video, must either a directory containing jpg files, or an .mp4 file.

optional arguments:
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
#Google Colab does not support cv2.imshow and the following method must be used instead
#from google.colab.patches import cv2_imshow


parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to the input video, must either a directory containing jpg files, or an .mp4 file.")
parser.add_argument("-o", "--output", type=str, help="Path to output directory.")
parser.add_argument("-d", "--display", action='store_true', help="Specify if results must be displayed in real time.")
args = parser.parse_args()

def run():
    # the freshold to see if we show as predicted
    THRESHOLD = 0.4

    # Initialize capture
    if isdir(args.input):
        cap = cv2.VideoCapture(join(args.input, "%06d.jpg"))
    else :
        cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    disp_width = round(0.5*width)
    disp_height = round(0.5*height)

    # initialize the network with Yolov3 weights and config
    net = cv2.dnn.readNet('yolov3_files/yolov3-spp.weights', 'yolov3_files/yolov3.cfg')

    #file containing class names
    with open('yolov3_files/coco.names', 'r') as f:
	    classes = f.read().splitlines()

    # initialize OpenCV's special multi-object tracker
    trackers = cv2.MultiTracker_create()

    no_detections = 0
    indexes = []
    tracker_labels = []
    tracker_confidences = []
    fps_arr = []

    # Parameters for info displaying
    FONT = cv2.FONT_HERSHEY_PLAIN
    green = (0, 255, 0)
    thickness = 1
    font_size = 1.5
    
    img_ctr = 0
    
    while True:
        _, img = cap.read()

        if img is None:
            break

        height, width, _ = img.shape

        frame_start_time = time.time()

        if(img_ctr % args.det_freq == 0):
            
            trackers = cv2.MultiTracker_create()
            tracker_labels = []
            tracker_confidences = []

            blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

            # set input from blob to network
            net.setInput(blob)

            # get output layer names
            output_layers_names = net.getUnconnectedOutLayersNames()
            # obtain outputs in output layer
            layerOutputs = net.forward(output_layers_names)

            boxes  = []
            confidences = []
            class_ids = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    # extract highest score
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > THRESHOLD:

                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        # width
                        w = int(detection[2] * width)
                        # height
                        h = int(detection[3] * height)

                        # position of upper left corner
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        box = (x, y, w, h)
                        boxes.append(box)
                        confidences.append((float(confidence)))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4).flatten()

            if len(indexes) > 0:
                for i in indexes:
                    x, y, w, h = boxes[i]
                    kcf_tracker = cv2.TrackerKCF_create()
                    trackers.add(kcf_tracker, img, boxes[i])
                    tracker_labels.append(str(classes[class_ids[i]]))
                    tracker_confidences.append(str(round(confidences[i], 2)))

        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked
        (_, boxes) = trackers.update(img)

        # loop over the bounding boxes and draw then on the frame
        for i in range(len(boxes)):
            box = boxes[i]
            (x, y, w, h) = [int(v) for v in box]

            no_detections += len(box)

            label = tracker_labels[i]
            confidence = tracker_confidences[i]
            #color = colors[i % 100]

            cv2.rectangle(img, (x, y), (x+w, y+h), green, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), FONT, font_size, green, thickness)
        
        # Time for performing detection and write the results on the frame.
        FPS = round(1.0 / (time.time() - frame_start_time), 2)
        fps_arr.append(FPS)
        cv2.putText(img, str(FPS) + " FPS" , (5, 40), FONT, 3, green, 2)
        image_name = "img"+str(img_ctr)+".jpg"

        if(args.display):
            cv2.imshow("SSD and MOSSE", cv2.resize(img, (disp_width, disp_height)))
            #cv2_imshow(cv2.resize(image_np, (disp_width, disp_height)))
            key = cv2.waitKey(1)
            if key==27:
                break
        if(not args.output is None):
            cv2.imwrite(join(args.output, image_name), cv2.resize(img, (1200, 700)))
        
        img_ctr += 1

    AVG_FPS = np.mean(fps_arr)
    print("Average FPS :", AVG_FPS)
    print("Number of detections: ", no_detections)
    cap.release()
    cv2.destroyAllWindows()
    print("Finished")

if __name__ == '__main__':
    run()
