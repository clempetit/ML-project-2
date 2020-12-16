import cv2
import numpy as np
import os
import argparse
import time


 # the freshold to see if we show as predicted
THRESHOLD = 0.4
 # font for the pictures
FONT = cv2.FONT_HERSHEY_PLAIN

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

with open('coco.names', 'r') as f:
	classes = f.read().splitlines()

 # capture video
 #cap = cv2.VideoCapture('street.mp4')

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Path to the input video, must either a directory containing jpg files, or an .mp4 file.")
args = parser.parse_args()

 # Initialize capture
if os.path.isdir(args.input):
    cap = cv2.VideoCapture(os.path.join(args.input, "%06d.jpg"))
else:
    cap = cv2.VideoCapture(args.input)

img_ctr = 0
indexes = []
tracker_labels = []
tracker_confidences = []


# colors and parameters for the frames
colors = np.random.uniform(0, 255, size=(100, 3))
green = (0, 255, 0)
thickness = 2
font_size = 2
no_detections = 0
fps_arr = []

while True:
	_, img = cap.read()

	if img is None:
		break

	height, width, _ = img.shape

	frame_start_time = time.time()

	if(img_ctr % 10 == 0):

		trackers = cv2.MultiTracker_create()
		tracker_labels = []
		tracker_confidences = []

		blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

		# shows the blobs that our algorithm processes
		# for b in blob:
		# 	for n, img_blob in enumerate(b):
		# 		cv2.imshow(str(n), img_blob)

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
	(success, boxes) = trackers.update(img)

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

	FPS = round(1.0 / (time.time() - frame_start_time), 2)
	fps_arr.append(FPS)
	cv2.putText(img, str(FPS) + " FPS" , (5, 40), FONT, 3, green, 2)
	image_name = "img"+str(img_ctr)+".jpg"

	cv2.imwrite(os.path.join("./results_kcf", image_name), cv2.resize(img, (1200, 700)))
	#cv2.imshow('Image', cv2.resize(img, (1200, 700)))
	# cv2.imshow('Image', img)
	key = cv2.waitKey(1)

	# if escape break
	if key==27:
		break

	img_ctr += 1

AVG_FPS = np.mean(fps_arr)
print("Average FPS :", AVG_FPS)
print("Number of detections: ", no_detections)
cap.release()
cv2.destroyAllWindows()

