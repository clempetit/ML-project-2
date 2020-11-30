'''
put the path of folder containing images preceded by --version
>>> python multi_object_tracking.py --video /Users/clementpetit/Desktop/img1/

press s, draw rectangle with mouse, and press enter

press q to quit
'''

# import the necessary packages
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os
import time

from os.path import isfile, join

#path = "/Users/clementpetit/Desktop/img1"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}
# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
	##vs = cv2.VideoCapture(args["video"])
    files = sorted([f for f in os.listdir(args["video"]) if isfile(join(args["video"], f))])
i = 0

det_file = open("/Users/clementpetit/Desktop/det.txt","r")
det_lines = det_file.readlines()
split_det_lines = [line.split(',') for line in det_lines]
for j in range(len(split_det_lines)):
    box =  split_det_lines[j]
    split_det_lines[j] = [float(elem) for elem in box]

frame_boxes_dict = {}
for box in split_det_lines:
    frame_boxes_dict.setdefault(box[0], []).append(box[2:6])

# loop over frames from the video stream
while True:
    filename=args["video"] + files[i]
    i = (i+1) % len(files)
    frame = cv2.imread(filename)
    if(i%30 == 0):
        print(i)
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    ##frame = vs.read()
    ##frame = frame[1] if args.get("video", False) else frame
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster)
    #frame = imutils.resize(frame, width=1000)

    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    (success, boxes) = trackers.update(frame)
    # loop over the bounding boxes and draw then on the frame
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for box in frame_boxes_dict.get(i+1):
        (x, y, w, h) = [round(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box)

        # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

    time.sleep(0.5)
'''        
# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()
'''
# close all windows
cv2.destroyAllWindows()