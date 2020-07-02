import argparse
import tensorflow as tf
import time
import cv2
import mss
import numpy as np

from tqdm import trange
from utils.config import Config
from model import ICNet, ICNet_BN

model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}

# Choose dataset here, but remember to use `script/downlaod_weight.py` first
dataset = 'cityscapes'
filter_scale = 1
    
class InferenceConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)
    
    # You can choose different model here, see "model_config" dictionary. If you choose "others", 
    # it means that you use self-trained model, you need to change "filter_scale" to 2.
    model_type = 'trainval'

    # Set pre-trained weights here (You can download weight from Google Drive) 
    model_weight = './model/cityscapes/icnet_cityscapes_trainval_90k.npy'
    
    # Define default input size here
    INFER_SIZE = (256, 512, 3)

cfg = InferenceConfig(dataset, is_training=False, filter_scale=filter_scale)
cfg.display()


# Create graph here 
model = model_config[cfg.model_type]
net = model(cfg=cfg, mode='inference')

# Create session & restore weight!
net.create_session()
net.restore(cfg.model_weight)

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 640, "height": 480}

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        screen = np.array(sct.grab(monitor))
        screen = np.flip(screen[:, :, :3], 2)
        capture = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        image = cv2.resize(screen, (cfg.INFER_SIZE[1], cfg.INFER_SIZE[0]))
        results1 = net.predict(image)
        overlap_results1 = (0.5 * image + 0.5 * results1[0])
        overlap_results1 = overlap_results1/255.0

        print("fps: {}".format(1 / (time.time() - last_time)))

        # Display the picture
        cv2.imshow("OpenCV/Numpy normal", capture)
        cv2.imshow("NN", overlap_results1)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break