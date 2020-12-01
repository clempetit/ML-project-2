""" TFRecord generator
Generates tfrecord from MOT17 folders (containing video frames) and det.txt file (containing detections coordinates for each frame)

usage: create_tfrecord.py [-h] [-i VIDEOS_DIR] [-o OUTPUT_PATH] [-l LABELS_PATH] [-f FREQUENCY]

optional arguments:
  -h, --help            show this help message and exit
  -f FREQUENCY
            Gap between images to be kept in each video

required arguments:
  -i VIDEOS_DIR
            Path to the folder containing MOT17 folders (where the input images and det.txt files are stored)
  -o OUTPUT_PATH
            Path of output TFRecord (.record) file.
  -l LABELS_PATH, --labels_path LABELS_PATH
            Path to the labels (.pbtxt) file.

"""

import os
import glob
import io
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument("-i", "videos_dir", type=str, help="Path to the folder containing MOT17 folders (where the input images and det.txt files are stored)")
parser.add_argument("-o", "output_path", type=str, help="Path of output TFRecord (.record) file.")
parser.add_argument("-l", "labels_path", type=str, help="Path to the labels (.pbtxt) file.")
parser.add_argument("-f", "--frequency", type=int, default=10, help="Gap between images to be kept in each video")
args = parser.parse_args()
                    

def create_boxes_dict(video_path):
    '''Creates a dictionary of boxes detected in every frame (only for frames that are to be selected).

    Parameters:
    ----------
    video_path : str
        The path to video MOT17 folder
    Returns
    -------
    frame_boxes_dict : dictionary
    '''

    frame_boxes_dict = {}

    det_path = os.path.join(video_path, "det/det.txt")
    with open(det_path, "r") as det_file :
        det_lines = det_file.readlines()

    for line in det_lines:
        parsed_line = [round(float(elem)) for elem in line.split(",")]
        frame_index = parsed_line[0]
        if((frame_index-1) % args.frequency == 0):
            # append x,y,w,h to dict
            frame_boxes_dict.setdefault(frame_index, []).append(parsed_line[2:6])
    
    return frame_boxes_dict


def create_tf_example(boxes, image_path, image_name):
    '''Creates a tf example for the given image and its detection boxes

    Parameters:
    ----------
    boxes : array
        The array containing the coordinates (x,y,w,h) of each box
    image_path : str
        The path to the image file
    image_name : str
        The image file name
    Returns
    -------
    tf_example
    '''
    
    category_index = label_map_util.create_category_index_from_labelmap(args.labels_path)

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = image_name.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for box in boxes:
        xmin, ymin, w, h = box
        xmax = xmin + w
        ymax = ymin + h
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(category_index[1]["name"].encode('utf8'))
        classes.append(category_index[1]["id"])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):

    writer = tf.python_io.TFRecordWriter(args.output_path)

    videos = os.listdir(args.videos_dir)
    for video in videos:
        if(not video.startswith("MOT17-")):
            continue

        video_path = os.path.join(args.all_vid_path, video)
        frame_boxes_dict = create_boxes_dict(video_path)
        
        # select images and write corresponding tf examples to tfrecord
        images_dir_path = os.path.join(args.all_vid_path, video, "img1", "")
        images = sorted(os.listdir(images_dir_path))
        for image_name in images:
            # image names of the form "000001.jpg", with always less than 10000 images per video
            if image_name[-4:] == ".jpg":
                image_id = int(image_name[-8:-4])
                if((image_id-1) % args.frequency == 0):
                    boxes = frame_boxes_dict.get(image_id)
                    # rename image to avoid conflict between image names of different videos
                    new_image_name = str(int(video[6:8])) + image_name[-8:]

                    old_image_path = os.path.join(images_dir_path, image_name)
                    new_image_path = os.path.join(images_dir_path, new_image_name)
                    os.rename(old_image_path + image_name, new_image_path)

                    tf_example = create_tf_example(boxes, new_image_path, new_image_name)
                    writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecord file: {}'.format(args.output_path))

if __name__ == '__main__':
    tf.app.run()
