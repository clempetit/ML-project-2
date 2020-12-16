# Machine Learning Project 2 : Multi-object detection and tracking

Please note that the repo contains all the data that we used (lots of .jpg files and a small .mp4 video), and weighs around 800MB. Hence the cloning may take a few minutes to complete.

## Overview of files

* In the top-level directory, you will find the different scripts that we wrote for this project :
    * `create_tfrecord.py`: A program that generates TFRecords from MOT Challenge data.
    * `create_config.py`: A program that modifies a pre-trained model's default pipeline.config, in order to adapt it to our own model.
    * `demo_ssd_mosse.py`: A demo program that runs SSD detection with MOSSE tracking on a given input video.
    * `demo_ssd_deepsort.py`: A demo program that runs SSD detection with DeepSORT tracking on a given input video.
    * `demo_yolo_kcf.py`: A demo program that runs YOLOv3 detection with KCF tracking on a given input video.
* The `tensorflow-models` package contains the code of the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
* The `deep_sort` package contains the official implementation of DeepSORT tracking from [Nicolas Wolkje](https://github.com/nwojke/deep_sort) and modified by [pythonlessons](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3/tree/master/model_data) for Tensorflow V2 compatibility, and the use of a trained DeepSORT model.
* The `images` folder contains train and test videos from [MOT Challenge]{https://motchallenge.net}. For each video the frames are located in the subfolder `img1` and the detections file `det.txt` in the subfolder `det`.
* The `training` folder contains the data we need for training the model.
    * The folder `pre-trained-models` contains the pre-trained MobileNet SSD model. Others pre-trained models can be downloaded from the [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
    * The `tf-records` folder contains our single-entry label map, and is intended to contain the TFRecords generated by `create_tfrecord.py`.
    * The `trained-model` folder contains the pipeline.config created by `create_config.py`, along with the checkpoint files. Initially, we included a checkpoint-6 from a training that we did ourselves on Google Colab, but you are free to empty this whole folder and redo the entire training by following the [Training](#Training "Goto Training") section.
* The `yolov3_files` folder contains the files needed to run demo_yolo_kcf.py, except for the yolov3 weights that can be downloaded at this [link](https://pjreddie.com/media/files/yolov3-spp.weights).
* `People.mp4` can be used for the demo.
* `Project-2-ML.ipynb` is a notebook you can run on Google Colab. All our work can be simply reproduced with this notebook.


## Installation :

If you are not running on macOS, some steps of the following configuration may fail. If you encounter problems or simply want to avoid all the configuration on your computer, you can simply run our Google Colab notebook available at this [link](https://colab.research.google.com/drive/1GQSJPOlUovb52Ol1ssBpKHHROjL_VjwP#scrollTo=LtZz188u0YU_), and go through the steps. This notebook clones and uses a github repo that is an exact copy of this one, but that we set to public (cloning private repos on Colab is quite tricky). We also provided a copy of the notebook in this repo, as "Project_2_ML.ipynb".

Furthermore, Google Colaboratory allows to use powerful CPUs and GPUs, which makes the training and demos faster. If you run the notebook on Google Colab, in the menu, go in Edit > Notebook settings, and select GPU in the "Hardware accelerator" field.

Note that in order to use GPUs from your own device when running the code in local, some extra setup is required for GPU support, which we did not describe here.

You will need to run the protocol buffer (protobuf) compiler protoc to generate some python files from the Tensorflow Object Detection API.
```bash
brew install protobuf
```

Create a new virtual environment with anaconda.
```bash
conda create -n ML_Project2_MOT pip python=3.8
conda activate ML_Project2_MOT
```

install tensorflow
```bash
pip install tensorflow
```

Install everything else that is required by the object-detection API. Please start by changing your directory to tensorflow_models/research :
```bash
cd tensorflow_models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
pip install opencv-contrib-python -U
```

You can test the installation with the following command (still from the tensorflow-models/research directory).
```bash
python object_detection/builders/model_builder_tf2_test.py
```
After a little while, it should output something like the following :
```bash
----------------------------------------------------------------------
Ran 20 tests in 68.510s

OK (skipped=1)
```

WARNING : cv2.MultiTracker_create() will be used for tracking, and is present only in the package opencv-contrib-python (not in opencv-python, that may be installed as well by default).
If an error occurs while calling MultiTracker_create(), uninstall opencv-python and opencv-contrib-python (if it was already installed), and reinstall opencv-contrib-python only.
```bash
pip uninstall opencv-contrib-python
pip uninstall opencv-contrib-python
pip install opencv-contrib-python
```

## Training

We included in the repo a trained model (see ckpt-6 in the /training/trained-model directory). If you want to run the demo directly, you can skip this section and go to the next section "Demo".

If you want to run the training, please start by deleting every file under training/trained-model.

Our .py scripts are documented and indications about the arguments is present in the respective files.

First, we need to convert the train and test data provided by MOT challenge into TFRecords. Please change your directory to the top-level directory (where our scripts are located).
```bash
python create_tfrecord.py images/train/ training/TFRecords/train.record training/TFRecords/label_map.pbtxt
python create_tfrecord.py images/train/ training/TFRecords/test.record training/TFRecords/label_map.pbtxt
```

Then, we need to create a customized pipeline.config file for our specific training. We will take the default config of the pre-trained model and modify it adequately. The batch size is 4 by default, you can set it to a higher value (e.g. 8 or 16) with flag `-b` if you are using Google Colab with GPUs.

```bash
python create_config.py training/pre-trained-models/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 training/TFRecords/label_map.pbtxt training/TFRecords training/trained-model
```

Now we are ready to run the training. This script is provided by the Tensorflow object detection API. Depending on the number of steps that you specify (in the following it is set to 5000), and the batch size (specified in the pipeline configuration above), the training may take some time to complete, especially if you are using only CPU. The program will first print some warnings that can be ignored, and after a little while, it will start to print the progression, at every 100 training steps.
```bash
python tensorflow_models/research/object_detection/model_main_tf2.py --model_dir training/trained-model/ --pipeline_config_path training/trained-model/pipeline.config --num_train_steps 5000
```
Note that reducing the number of steps would of course make the model less accurate, and conversely, too high a value could result in overfitting. After several tries, we estimated that 5000 steps was a good bet.

## Demo
Please change your directory to the top-level directory (where our scripts are located).
In order to test our trained detection model, with either MOSSE or DeepSORT tracking, you can simply run one of the two following scripts. The scripts take as input either a .mp4 file or a folder containing .jpg files. You have the choice to save the results with the `-o` flag and/or to visualize them in real time with the `-d` flag (please note that the latter does not work on Google Colab). Some other options are documented inside the scripts.

SSD detection with MOSSE tracking (this script takes in addition, as argument, the `trained-model` directory) :
```bash
# save the results in an empty results/ directory :
rm -r results; mkdir results
python demo_ssd_mosse.py People.mp4 training/trained-model/ -o results/

# display the images in real time. Does not work on Google Colab !
python demo_ssd_mosse.py People.mp4 training/trained-model/ -d
```

SSD detection with DeepSORT tracking (this scrit takes in addition, as argument, the path to a DeepSORT model that we downloaded from the [pythonlessons](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3/tree/master/model_data) repo) :
```bash
# save the results in an empty results/ directory :
rm -r results; mkdir results
python demo_ssd_deepsort.py People.mp4 training/trained-model/ /deep_sort/mars-small128.pb -o results/

# display the images in real time. Does not work on Google Colab !
python demo_ssd_deepsort.py People.mp4 training/trained-model/ /deep_sort/mars-small128.pb -d
```

Finally, you can also have a look at a YOLO v3 demo (with KCF tracking), that we had previously written in order to have a first glimpse at object detection, and that we used later to make comparisons with our own SSD model. In order to run this script, you will need first to download the weights [yolov3-spp.weights](https://pjreddie.com/media/files/yolov3-spp.weights) from pjreddie.com (the file weighs about 200MB, which was too big to put inside this repo) and put the file inside the folder yolov3_files :
```bash
# save the results in an empty results/ directory :
rm -r results; mkdir results
python demo_yolo_kcf.py People.mp4 -o results/

# display the images in real time. Does not work on Google Colab !
python demo_yolo_kcf.py People.mp4 -d
```