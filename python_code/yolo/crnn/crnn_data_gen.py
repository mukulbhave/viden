import numpy as np
import keras
import cv2


class DataGenerator(keras.utils.Sequence):

    def getTrainData(self,dataset,indexes):
        train_img_list = dataset['train_img_list'][indexes]
        train_padded_txt_list = dataset['train_padded_txt_list'][indexes]
        train_label_length_list = dataset['train_label_length_list'][indexes]
        train_input_length_list = dataset['train_input_length_list'][indexes]
        return (train_img_list,train_padded_txt_list,train_label_length_list,train_input_length_list)
    
    def getValData(self,dataset,indexes):
        val_img_list = dataset['val_img_list'][indexes]
        val_padded_txt_list = dataset['val_padded_txt_list'][indexes]
        val_label_length_list = dataset['val_label_length_list'][indexes]
        val_input_length_list = dataset['val_input_length_list'][indexes]
        return (val_img_list,val_padded_txt_list,val_label_length_list,val_input_length_list)
        
    'Generates data for Keras'
    def __init__(self, hdf5_dataset,data_set_size,is_train=1, batch_size=256,  shuffle=True):
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
        images , padded_txt , label_len,input_len = self.getTrainData(self.hdf5_dataset,indexes) if self.is_train ==1 else self.getValData(self.hdf5_dataset,indexes)
       
        images_list = [ cv2.imdecode(x, cv2.IMREAD_COLOR) for x in images]
        images_list = [np.array(image, dtype=np.float) for image in images_list]
        images_list = [image/255. for image in images_list]
        images_list = np.array(images_list)
        images_list = np.mean(images_list,axis=3,keepdims=True)
        padded_txt = np.array (padded_txt)
        label_len = np.array(label_len)
        input_len = np.array(input_len)
        # Store sample
        X = [images_list,padded_txt,input_len,label_len]
        y = np.zeros(images_list.shape[0])
        
        return X, y
    
    
    
    
