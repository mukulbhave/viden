# Load the yolo model and crnn model for prediction
# For an input image detect title ,author and price. We will get at most 6 bounding boxes
# the bounding boxes in set of (title,author,price)
# Take out the region from original image for each bounding box
# Predict the text in it. Compare with the label
# calculate precision and recall
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

argparser = argparse.ArgumentParser(
    description="evaluate the model.")
argparser.add_argument(
    '-i',
    '--input_path',
    
    help="path to test image file")
    
argparser.add_argument(
    '-o',
    '--out_path',
    
    help="path to output predicted  images")
    
argparser.add_argument(
    '-c',
    '--classes_path',
    help='yolo path to classes file, defaults to classes.txt',
    default=os.path.join('.', 'classes.txt'))

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))





def _main(args):
   
    input_path = args.input_path
    out_path = args.out_path
    classes_path =args.classes_path
    outF = open(out_path+"\prediction.txt", "w")
    
    # model to be used at test time
    act_model = create_crnn_model(train=False)
    # load the saved best model weights
    act_model.load_weights('viden_trained_models\\viden_crnn_14May2021.hdf5')

    class_names = get_classes(classes_path)
    anchors = YOLO_ANCHORS 
    yolo_model_body, yolo_model = create_model(anchors, class_names)
    yolo_model_body.load_weights('viden_trained_models\\viden_yolo_14May2021.h5')

    yolo_outputs = yolo_head(yolo_model_body.output, anchors, len(class_names))
    yolo_input_image_shape = K.placeholder(shape=(2, ))
        
    boxes, scores, classes = yolo_eval(
            yolo_outputs, yolo_input_image_shape,max_boxes=6, score_threshold=.7, iou_threshold=0.5)
    config = ('-l eng --oem 1 --psm 3')  
    time_lst=[]
    orig_img_name_lst=[]
    for img_path in glob.iglob(input_path+"*.jpg"):
        orig_img_name=''
        print("Evaluating :"+img_path)
        start_time = time.time()
        pred_boxes,pred_box_classes = get_bounding_boxes(yolo_model_body,boxes, scores, classes,yolo_input_image_shape,img_path,class_names,out_path,save=True)
        outF.write("File Name:"+img_path)
        outF.write("\n")
        for i, box in list(enumerate(pred_boxes)):
            box_class = class_names[pred_box_classes[i]]
           # box = pred_boxes[i]
            top, left, bottom, right = box
            image = PIL.Image.open(img_path)
            pred_obj_img = image.crop((left,top,right,bottom))
            txt=get_text(act_model,pred_obj_img,box_class)
            
            outF.write(box_class+":"+txt)
            outF.write("\n")
            print(txt)
            _,tail = ntpath.split(img_path)
            orig_img_name =os.path.splitext(tail)[0]
            
            pred_obj_img.save(out_path+"\\"+txt+"_"+str(i)+"_"+box_class+".jpg")
        end_time = time.time()
        time_lst.append(end_time-start_time)
        orig_img_name_lst.append(orig_img_name)
        outF.write("\n")
    
    outF.close()
    plot_time(out_path,orig_img_name_lst,time_lst)
    

def plot_time(out_path,names,time_taken):
    avg_time = np.sum(time_taken) / len(time_taken)
    plt.plot(names,time_taken,linewidth=2, color='navy')
    plt.xlabel("Images", fontsize=12, fontweight='bold')
    plt.ylabel("Evaluation Time", fontsize=12,fontweight='bold')
    plt.title("Evaluation time(sec) per Image (Avg:{0:0.2f})".format(avg_time), fontsize=18, fontweight="bold")
    plt.ylim( 0,3)
    fig = plt.gcf()
    fig.set_size_inches(20, 8.5)
    file = out_path+'\\eval_time.jpg'
    plt.show()
    fig.savefig(file)
    
def get_bounding_boxes(yolo_model_body,boxes, scores, classes,yolo_input_image_shape,img_path,class_names,out_path,save=True):
    image_shape = (416, 416) 
    image = PIL.Image.open(img_path)
    
    resized_image =image.resize(tuple(image_shape), PIL.Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)
    
    sess = K.get_session()
    out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model_body.input: image_data,
                yolo_input_image_shape: [image_data.shape[1], image_data.shape[2]],
                K.learning_phase(): 0
            })
    
    _,tail = ntpath.split(img_path)
    tail =os.path.splitext(tail)[0]
    print("file Name:"+tail)
    orig_img_name = out_path+"\\"+tail+".jpg"
    
    
   # image_with_boxes = draw_boxes(image_data[0], out_boxes, out_classes, class_names, out_scores,orig_img_name)
    
    # Convert pred on 416 to actual image size
    resized_boxes = out_boxes/416
    
    w,h = image.size
    box_resize_dim = [h,w,h,w]
    resized_boxes = resized_boxes * box_resize_dim
    orig_image_data = np.array(image, dtype='float32')
    orig_image_with_boxes = draw_boxes(orig_image_data, resized_boxes, out_classes, class_names, out_scores,orig_img_name)
	
    # Save the image:
    if save:
        orig_image = PIL.Image.fromarray(orig_image_with_boxes)
        orig_image.save(orig_img_name)
    
    return resized_boxes,out_classes

def get_text(act_model,pil_image,class_name):
    img = pil_image.resize((128, 32), Image.BICUBIC)
    img = np.array(img) /255;
    img = np.sum(img, axis=2,keepdims=True)
    img = np.expand_dims(img , axis = 0)

    prediction = act_model.predict(img)
    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],greedy=False)[0][0])
    x = out[0]
    s=''
    for p in x:  
        if int(p) != -1:
            s += char_list[int(p)]
    return s
                                               
#out_scores, out_boxes, out_classes = predict(sess, "3.jpg")
if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)             
                         