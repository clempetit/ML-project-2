
# Machine Learning Project 2 : Multi-object detection and tracking

If you are not running on macOS, some steps of the following configuration may fail. If you encounter problems or simply want to avoid all the configuration on your computer, you can simply run the attached notebook "Project_2_ML.ipynb" on Google Collaboratory and go through the steps.
Furthermore, Google Collaboratory allows to use powerful CPUs and GPUs, which makes the training and demos faster. If you run the notebook on Google Collab, in the menu, go in Edit > Notebook settings, and select GPU in the "Hardware accelerator" field.

## Installation on Mac OS: 

You will need to run the protocol buffer (protobuf) compiler protoc to generate some python files from the Tensorflow Object Detection API.
```bash
brew install protobuf
```

Create a new virtual environment with anaconda.
```bash
conda create -n tensorflow pip python=3.8
```

install tensorflow
```bash
pip install tensorflow
```

Install evrything else that is required by the object-detection API.
```bash
cd tensorflow_models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
pip install opencv-contrib-python -U
```

WARNING : cv2.MultiTracker_create() will be used for tracking, and is present only in package opencv-contrib-python (not in opencv-python, that may be installed as well by default).
If an error occur while calling MultiTracker_create(), uninstall opencv-python and opencv-contrib-python, and reinstall opencv-contrib-python only.
```bash
pip uninstall opencv-contrib-python
pip install opencv-contrib-python
```

## Training

We included in the repo a trained model (see ckpt-6 in the /training/trained-model directory). If you want to run the demo directly, you can skip this section and go to the next section "Demo".

If you want to run the training, please start by deleting every the file under training/trained-models.

For the different .py scripts to be run, a description of the arguments is present in the respective files.

First, we need to convert the train and test data provided by MOT challenge into TFRecords.
```bash
!python create_tfrecord.py images/train/ training/TFRecords/train.record training/TFRecords/label_map.pbtxt -f 10
!python create_tfrecord.py images/train/ training/TFRecords/test.record training/TFRecords/label_map.pbtxt -f 10
```

Then, we need to create a customized pipeline.config file for our specific traing. We will take the default config of the pre-trained model and modify it adequately.

```bash
!python create_config.py training/pre-trained-model/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 training/TFRecords/label_map.pbtxt training/TFRecords training/trained-model
```

Now we are ready to run the training. Depending on the number of steps that you specify (in the following it is set to 5000), and the batch size (4 by default), the trainig may take some time. The program prints the progression at every 100 steps.
```bash
!python tensorflow_models/research/object_detection/model_main_tf2.py --model_dir training/trained-model/ --pipeline_config_path training/trained-model/pipeline.config --num_train_steps 5000
```

## Demo

In order to test our model, simply run the following script. You have the choice to save the results with the -o flag and/or to visualize them in real time with the -d flag.

```bash
# saves the results in results/ directory :
!python demo_ssd_mosse.py People.mp4 training/trained-model/ -o results/

# display the images in real time.
!python demo_ssd_mosse.py People.mp4 training/trained-model/ -d
```