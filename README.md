
# Machine Learning Project 2 : Multi-object detection and tracking

If you are not running on macOS, some steps of the following configuration may fail. If you encounter problems or simply want to avoid the configuration steps, you can simply run the attached notebook "Project_2_ML.ipynb".

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