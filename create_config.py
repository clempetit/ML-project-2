'''Generates customized pipeline config for SSD model training

usage : create_config.py [PRE_TRAINED_MODEL] [LABEL_MAP] [TFRECORD] [TRAINED_MODEL] [BATCH_SIZE]

optional arguments:
    -b BATCH_SIZE
        Batch size for the training.

required arguments:
    PRE_TRAINED_MODEL
        Path of the pre-trained model's directory, containing the config file and the training checkpoint 0.

    LABEL_MAP
        Path of the label map file.

    TFRECORD
        Path of the directory containing train and test TFrecords.

    TRAINED_MODEL
        Path of the trained model's directory.
'''

import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

parser = argparse.ArgumentParser()
parser.add_argument("pre_trained_model", type=str, help="Path of the pre-trained model's directory.")
parser.add_argument("label_map", type=str, help="Path of the label map file.")
parser.add_argument("TFRecord", type=str, help="Path of the directory containing train and test TFrecords.")
parser.add_argument("trained_model", type=str, help="Path of the trained model's directory.")
parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size for the training.")
args = parser.parse_args()

CONFIG_PATH = os.path.join(args.pre_trained_model, "pipeline.config")
STARTING_CKPT_PATH = os.path.join(args.pre_trained_model, "checkpoint/ckpt-0")

def custom_config():
    custom_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
        default_config = f.read()
        text_format.Merge(default_config, custom_config)

    custom_config.model.ssd.num_classes = 1

    # This model will be used to perform detection
    custom_config.train_config.fine_tune_checkpoint_type = "detection"

    # Start training from a pretained model 
    custom_config.train_config.fine_tune_checkpoint = STARTING_CKPT_PATH

    # Paths to label map and tf record for training
    custom_config.train_input_reader.label_map_path= args.label_map
    custom_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(args.TFRecord, 'train.record')]

    # Paths to label map and tf record for evaluation
    custom_config.eval_input_reader[0].label_map_path = args.label_map
    custom_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(args.TFRecord, 'test.record')]

    # As we are using Google's GPUs (Tesla T4, P100...), large batch size will improve performances
    custom_config.train_config.batch_size = args.batch_size

    custom_config_str = text_format.MessageToString(custom_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(args.trained_model + '/pipeline.config', "wb") as f:                                                                                                                                                                                                                   
        f.write(custom_config_str)
    print("Successfully created the pipeline config.")

if __name__ == '__main__':
    custom_config()