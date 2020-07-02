# Implementation of Steering angle prediction and ICNet Semantic Segmentation

1. Download or clone this repo
2. Navigate to location of this repo on your hard disk using Anaconda Prompt using cd
3. To run steering + ICNet simultaneously: **python merge.py**

# In merge.py:

To modify size of input image to ICNet, change **line 35**, by default the value is:
INFER_SIZE = (256, 512, 3)

To modify part of the screen to capture, change **line 87**, by default it is:
monitor = {"top": 0, "left": 0, "width": 640, "height": 200}



For steering only: **python steer.py**
For ICNet only: **python icnet.py**