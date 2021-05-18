import argparse
import os
import fnmatch
import cv2
import h5py
import numpy as np
import string
import time,datetime
import keras
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint , TensorBoard,EarlyStopping
from PIL import Image
#from crnn_data_gen import DataGenerator
import tensorflow as tf

# char_list:   punctuation,digits,ascii_letters ,' '
# total number of our output classes: len(char_list)  
char_list = string.ascii_letters+string.digits
#char_list = string.ascii_letters+string.digits+string.punctuation+' '

argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to  'images' folder",
    default='C:\\dataset\\CRNN\\mnt\\ramdisk\\max\\train_90kDICT32.hdf5')


argparser.add_argument(
    '-rt',
    '--resume_train',
    help="Resume training if 1 else train from start the model again by loading prev trained weights",
    default='0')
    
def _main(args):
  tf.device('/gpu:1')
  data_path = args.data_path
  max_label_len = 31
  model = create_crnn_model(max_label_len=max_label_len)
  filepath="best_model.hdf5"
  if args.resume_train =='1':
    print("loading pretained weights")
    model.load_weights("best_model.hdf5")
    filepath="retrain_best_model.hdf5"
  log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  logging = TensorBoard(log_dir=log_dir)
  checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
  early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
  callbacks_list = [checkpoint,logging,early_stopping]

  batch_size = 256
  epochs = 50
 
  # training_img = np.array(training_img)
  # print(training_img[0].shape)
  # train_input_length = np.array(train_input_length)
  # train_label_length = np.array(train_label_length)

  # valid_img = np.array(valid_img)
  # valid_input_length = np.array(valid_input_length)
  # valid_label_length = np.array(valid_label_length)
  
  f = h5py.File(data_path,'r+')
  train_dataset = f['train']
  val_dataset = f['val']
  train_set_size= train_dataset.attrs['dataset_size']
  val_set_size = val_dataset.attrs['dataset_size']
  training_generator = DataGenerator(train_dataset, train_set_size,batch_size=batch_size)
  validation_generator = DataGenerator(val_dataset, val_set_size,batch_size=batch_size,is_train=0)
  
  model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    epochs=epochs,verbose = 1, callbacks = callbacks_list)
  # model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], 
                # y=np.zeros(len(training_img)), batch_size=batch_size, epochs = epochs,
                # validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length],
                # [np.zeros(len(valid_img))]), verbose = 1, callbacks = callbacks_list)
 

def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst

def create_crnn_model(train=True,max_label_len=0):
  # input with shape of height=32 and width=128 
  inputs = Input(shape=(32,128,1))
 
  # convolution layer with kernel size (3,3)
  conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
  # poolig layer with kernel size (2,2)
  pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
  conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
  pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
  conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
  conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
  # poolig layer with kernel size (2,1)
  pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
  conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
  # Batch normalization layer
  batch_norm_5 = BatchNormalization()(conv_5)
 
  conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
  batch_norm_6 = BatchNormalization()(conv_6)
  pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
  conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
  squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 
  # bidirectional LSTM layers with units=128
  blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.3))(squeezed)
  blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.3))(blstm_1)
 
  outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)
  if(train):
    #model to be used at training time
    labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
  
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
#lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = Adam(lr=0.0005))

    return model
  
  # model to be used at test time
  act_model = Model(inputs, outputs)
  return act_model

def ctc_lambda_func(args):
  y_pred, labels, input_length, label_length = args
  return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



def process_data(path):
 
  # lists for training dataset
  training_img = []
  training_txt = []
  train_input_length = []
  train_label_length = []
  orig_txt = []
 
  #lists for validation dataset
  valid_img = []
  valid_txt = []
  valid_input_length = []
  valid_label_length = []
  valid_orig_txt = []
 
  max_label_len = 0
 
  i =1 
  flag = 0
 
  for root, dirnames, filenames in os.walk(path):
 
    for f_name in fnmatch.filter(filenames, '*.jpg'):
        #print(f_name)
        # read input image and convert into gray scale image
        img = cv2.cvtColor(cv2.imread(os.path.join(root, f_name)), cv2.COLOR_BGR2GRAY)   
 
        # convert each image of shape (32, 128, 1)
        w, h = img.shape
        if h > 128 or w > 32:
            continue
        if w < 32:
            add_zeros = np.ones((32-w, h))*255
            img = np.concatenate((img, add_zeros))
 
        if h < 128:
            add_zeros = np.ones((32, 128-h))*255
            img = np.concatenate((img, add_zeros), axis=1)
        img = np.expand_dims(img , axis = 2)
        
        # Normalize each image
        img = img/255.
        
        # get the text from the image
        txt = f_name.split('_')[1]
        
        # compute maximum length of the text
        if len(txt) > max_label_len:
            max_label_len = len(txt)
            
           
        # split the 150000 data into validation and training dataset as 10% and 90% respectively
        if i%10 == 0:     
            valid_orig_txt.append(txt)   
            valid_label_length.append(len(txt))
            valid_input_length.append(31)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))
        else:
            orig_txt.append(txt)   
            train_label_length.append(len(txt))
            train_input_length.append(31)
            training_img.append(img)
            training_txt.append(encode_to_labels(txt)) 
        
        # break the loop if total data is 150000
        if i == 150000:
            flag = 1
            break
        i+=1
    if flag == 1:
        break
        
  # pad each output label to maximum text length

  train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value = len(char_list))
  valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = len(char_list))
  return (max_label_len,training_img,train_padded_txt,train_input_length,train_label_length,valid_img,valid_padded_txt,valid_input_length,valid_label_length)




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

if __name__ == '__main__':
  args = argparser.parse_args()
  _main(args)   
  
  

 
 

 
 