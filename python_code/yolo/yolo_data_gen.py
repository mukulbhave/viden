import numpy as np
import keras
from retrain_yolo import * #scale_data,get_detector_mask,YOLO_ANCHORS
from PIL import ImageOps
import numpy as np
import PIL
class DataGenerator(keras.utils.Sequence):

    def getTrainData(self,dataset,indexes):
        train_img_list = dataset['images'][indexes]
        train_boxes = dataset['boxes'][indexes]
        image_data, boxes = scale_data(train_img_list, train_boxes)
        return (image_data, boxes)
    
    def getValData(self,dataset,indexes):
        val_img_list = dataset['val_img_list'][indexes]
        val_padded_txt_list = dataset['val_padded_txt_list'][indexes]
        val_label_length_list = dataset['val_label_length_list'][indexes]
        val_input_length_list = dataset['val_input_length_list'][indexes]
        return (val_img_list,val_padded_txt_list,val_label_length_list,val_input_length_list)
        
    'Generates data for Keras'
    def __init__(self, hdf5_dataset,data_set_size,is_train=1, batch_size=8,  shuffle=True):
        'Initialization'
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hdf5_dataset = hdf5_dataset
        self.data_set_size = data_set_size
        self.indexes = np.arange(data_set_size)
        no_of_batches = int(np.floor(self.data_set_size / self.batch_size))
        self.no_of_batches =  no_of_batches if self.data_set_size % self.batch_size == 0 else no_of_batches+1
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.no_of_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if index == self.no_of_batches -1 :
            indexes = self.indexes[index*self.batch_size:]
        else :
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data , need to load data from file from training list
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        #Indexing elements must be in increasing order
        indexes.sort()
        # Initialization
        image_data, boxes = self.getTrainData(self.hdf5_dataset,indexes)
       
        detectors_mask, matching_true_boxes = get_detector_mask(boxes, YOLO_ANCHORS)
        # Store sample
        X = [image_data,boxes,detectors_mask,matching_true_boxes]
        y = np.zeros(image_data.shape[0])
        
        return X, y
    
