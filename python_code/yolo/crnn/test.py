import argparse
import os
import fnmatch
import cv2
import numpy as np
import string
import time
import ntpath
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from PIL import Image
import glob
from pathlib import Path
from train_crnn import create_crnn_model
from util import process_data
char_list = string.ascii_letters+string.digits
img_path="C:\\dataset\\viden_test\\test_plates2\\"



# model to be used at test time
act_model = create_crnn_model(train=False)
# load the saved best model weights
act_model.load_weights('best_model.hdf5')


for pathAndFilename in glob.iglob(img_path+"*.jpg"):
  print("predicting for:"+pathAndFilename)
# predict outputs on validation images
  # img = Image.open(pathAndFilename)
  # img = img.resize((128, 32), Image.BICUBIC)
  
  # img = np.array(img) /255;
  # img = np.sum(img, axis=2,keepdims=True)
  img , _,_,_ =process_data(pathAndFilename,"1_1")
  img = img/255.
  img = np.expand_dims(img , axis = 0)
  prediction = act_model.predict(img)

# use CTC decoder
  out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=False)[0][0])
  head, tail = ntpath.split(pathAndFilename)
  txt = tail.split('_')[1]
# see the results
  i = 0
  le= min(10,out.shape[1])
  print(out.shape)
  for x in out:
    print(txt)
    for p in range(0, le):  
        if int(x[p]) != -1:
            print(char_list[int(x[p])], end = '')       
    print('\n')
    i+=1
    
