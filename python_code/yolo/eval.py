
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)


import h5py
import io
import ntpath
from yad2k.utils.draw_boxes import draw_boxes
#from yolo_utils import *
from retrain_yolo import (create_model,get_classes)

argparser = argparse.ArgumentParser(
    description="evaluate the model.")
argparser.add_argument(
    '-d',
    '--data_path',
    
    help="path to test image data file (.jpg files) with trailing \ or /")
    
argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to classes.txt',
    default=os.path.join('.', 'classes.txt'))

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

image_shape = (416, 416) 

def _main(args):
    # Current directory
    img_path = args.data_path
    classes_path =args.classes_path
    print("processing path:"+img_path)
    class_names = get_classes(classes_path)
    anchors = YOLO_ANCHORS #read_anchors("model_data/yolo_anchors.txt")
     

    model_body, model = create_model(anchors, class_names)
    predict_draw(model_body, class_names, anchors, img_path, 
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=False)



def predict_draw(model_body, class_names, anchors, img_path, 
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
	
    head, tail = ntpath.split(img_path)
    image = PIL.Image.open(img_path)
    resized_image =image.resize(tuple(image_shape), PIL.Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    
    image_data /= 255.
    
    image_data = np.expand_dims(image_data, 0)
    print(image_data.shape)
    # model.load_weights(weights_name)
   # print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape,max_boxes=3, score_threshold=.7, iou_threshold=0.05)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    #for i in range(len(image_data)):
    out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data,
                input_image_shape: [image_data.shape[1], image_data.shape[2]],
                K.learning_phase(): 0
            })
    print('Found {} boxes for image.'.format(len(out_boxes)))
    print(out_boxes)

	# Plot image with predicted boxes.
    image_with_boxes = draw_boxes(image_data[0], out_boxes, out_classes, class_names, out_scores)
    
   
	# Save the image:
    if save_all or (len(out_boxes) > 0):
	    image = PIL.Image.fromarray(image_with_boxes)
	    image.save(os.path.join(out_path,tail+'.png'))

	# To display (pauses the program):
    plt.imshow(image_with_boxes, interpolation='nearest')
    plt.show()
	



#out_scores, out_boxes, out_classes = predict(sess, "3.jpg")
if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args) 