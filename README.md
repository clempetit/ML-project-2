On Mac OS: 

#install protobuf :
brew install protobuf

#create a new virtual environment with anaconda
conda create -n tensorflow pip python=3.8

#install tensorflow
pip install tensorflow

#install evrything required by the object-detection API
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
pip install opencv-contrib-python -U

Then cd in the root folder

Otherwise, run on google colab