import argparse
import cv2
import re
import numpy as np
import string
import PIL
import os,glob
import ntpath
import time
import matplotlib.pyplot as  plt
from PIL import Image
from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes
from retrain_yolo import (create_model,get_classes)

import keras.backend as K

from crnn.train_crnn import create_crnn_model
from crnn.crnn_data_gen import *

char_list = string.ascii_letters+string.digits
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))
class_names=['plate','no-plate']   
class CarNumberDetector:
    def __init__(self,viden_yolo_weights_path,viden_crnn_weights_path,classes_path,out_path):
        self.act_model = create_crnn_model(train=False)
        self.act_model.load_weights(viden_crnn_weights_path)# 'viden_trained_models\\viden_crnn_14May2021.hdf5')
        class_names = get_classes(classes_path)
        #print(class_names)
        self.out_path=out_path
        #self.anchors = YOLO_ANCHORS 
        self.yolo_model_body, self.yolo_model = create_model(YOLO_ANCHORS, class_names)
        self.yolo_model_body.load_weights(viden_yolo_weights_path)#'viden_trained_models\\viden_yolo_14May2021.h5')
        self.yolo_outputs = yolo_head(self.yolo_model_body.output, YOLO_ANCHORS, len(class_names))
        self.yolo_input_image_shape = K.placeholder(shape=(2, ))
        
        self.boxes, self.scores, self.classes = yolo_eval(
            self.yolo_outputs, self.yolo_input_image_shape,max_boxes=1, score_threshold=.7, iou_threshold=0.5)
        
    
    def extract_number(self,orig_image_url=None,image_array=None,save=False):
        """
        This is the primary method to detect number plate on car and fetch the number.
        image_array is the numpy array representing original car image
        method returns numpy array of image with bounding box ad the extracted car_number string
        """
        if( image_array is None and orig_image_url is None):
            raise ValueError("image array or url is required")
        if(orig_image_url is not None):
            image = PIL.Image.open(orig_image_url)
        else:
            image = PIL.Image.fromarray(image_array)
        pred_boxes,pred_box_classes,img_with_boxes = self.get_bounding_boxes(image)
        pred_txt=''
        
        for i, box in list(enumerate(pred_boxes)):
            box_class = class_names[pred_box_classes[i]]
           
            top, left, bottom, right = box
            
            pred_obj_img = image.crop((left,top,right,bottom))
            
            pred_txt=self.get_text(pred_obj_img)
        
        # Save the image:
        if save:
            time_param= int(round(time.time() * 1000))
            orig_img_name = self.out_path+"\\"+pred_txt+"-"+str(time_param)+".jpg"
            orig_image = PIL.Image.fromarray(img_with_boxes)
            orig_image.save(orig_img_name)
            
        return (img_with_boxes,pred_txt)
        
    
    def get_text(self,pil_image):
        img = pil_image.resize((128, 32), Image.BICUBIC)
        img = np.array(img) /255;
        img = np.sum(img, axis=2,keepdims=True)
        img = np.expand_dims(img , axis = 0)

        prediction = self.act_model.predict(img)
        # use CTC decoder
        out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],greedy=False)[0][0])
        x = out[0]
        le= min(10,out.shape[1])

        s=''
        for x in out:
            for p in range(0, le): 
                if int(x[p]) != -1:
                    s += char_list[int(x[p])]
               
        return s
        
    
    def get_bounding_boxes(self,image):
        image_shape = (416, 416) 
        
        resized_image =image.resize(tuple(image_shape), PIL.Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)
        
        sess = K.get_session()
        out_boxes, out_scores, out_classes = sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model_body.input: image_data,
                    self.yolo_input_image_shape: [image_data.shape[1], image_data.shape[2]],
                    K.learning_phase(): 0
                })
        
        
        # Convert pred on 416 to actual image size
        resized_boxes = out_boxes/416
        
        w,h = image.size
        box_resize_dim = [h,w,h,w]
        resized_boxes = resized_boxes * box_resize_dim
        orig_image_data = np.array(image, dtype='float32')
        orig_image_with_boxes = draw_boxes(orig_image_data, resized_boxes, out_classes, class_names, out_scores,"rand")
        
        return resized_boxes,out_classes,orig_image_with_boxes