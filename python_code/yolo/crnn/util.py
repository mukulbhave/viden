import argparse
import os,glob,fnmatch
import h5py
import numpy as np
import cv2
import string
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt



char_list = string.ascii_letters+string.digits
max_label_len = 31
val_percentage = 10
MAX_COUNT = 300000; #// dataset size 
parser = argparse.ArgumentParser(
    description='Text detection dataset to HDF5.')
parser.add_argument(
    '-p',
    '--path_to_dataset',
    help='path to dataset directory',
    default='C:\\dataset\\CRNN\\mnt\\ramdisk\\max\\90kDICT32px\\')

parser.add_argument(
    '-ot',
    '--train_file_name',
    help='name of output file',
    default='C:\\dataset\\CRNN\\mnt\\ramdisk\\max\\train_crnn_tiny.hdf5')

parser.add_argument(
    '-ov',
    '--val_file_name',
    help='name of output file',
    default='C:\\dataset\\CRNN\\mnt\\ramdisk\\max\\val_90kDICT32.hdf5')
    
parser.add_argument(
    '-r',
    '--read_data',
    help='1 if sample data should be read as well else 0 default',
    default='0')


def _main(args):
  input_path = args.path_to_dataset
  out_put_path = args.train_file_name
  read_data = args.read_data
  create_data_file(input_path,out_put_path)
  if read_data == '1':
    read_data_file(out_put_path)


#Create mat iterates ovethe file name pattern in dir and creates a hdf5 data file with name as output_path
def create_data_file(input_path,output_path) :
    """ A helper function to create  data in mat format  """
    h5file = h5py.File(output_path, 'w')
    train_group = h5file.create_group('train')
    val_group = h5file.create_group('val')
    
    uint8_dt = h5py.special_dtype(
        vlen=np.dtype('uint8'))
    dt = h5py.special_dtype(vlen=str)
    max_len = max_label_len
    # lists for training dataset
    train_img_list = train_group.create_dataset(
        'train_img_list', shape=(MAX_COUNT, ),dtype=uint8_dt)
    train_padded_txt_list = train_group.create_dataset(
        'train_padded_txt_list', shape=(MAX_COUNT,max_len ))
    train_txt_list = train_group.create_dataset(
        'train_txt_list', shape=(MAX_COUNT, ),dtype=dt)
    train_label_length_list = train_group.create_dataset(
        'train_label_length_list', shape=(MAX_COUNT, ))
    train_input_length_list = train_group.create_dataset(
        'train_input_length_list', shape=(MAX_COUNT, ))
        
        
    val_img_list = val_group.create_dataset(
        'val_img_list', shape=(MAX_COUNT, ),dtype=uint8_dt)
    val_padded_txt_list = val_group.create_dataset(
        'val_padded_txt_list', shape=(MAX_COUNT,max_len ))
    val_txt_list = train_group.create_dataset(
        'val_txt_list', shape=(MAX_COUNT, ),dtype=dt)
    val_label_length_list = val_group.create_dataset(
        'val_label_length_list', shape=(MAX_COUNT, ))
    val_input_length_list = val_group.create_dataset(
        'val_input_length_list', shape=(MAX_COUNT, ))
    
    val_count_check = round(100 / val_percentage) 
    
    count = 0
    val_count = 0
    train_count = 0
    val_idx = 1
    flag =0
    for root, dirnames, filenames in os.walk(input_path):
        print("processing:  "+root)
        for f_name in fnmatch.filter(filenames, '*.jpg'):
            label_img = f_name.replace(".jpg",'')
            train_img,train_padded_txt,txt_length,orig_txt= process_data(os.path.join(root, f_name),label_img) 
            
            if txt_length is None:
                print( "Skiping as failed to load img:"+f_name)
               
                continue
            
            img_str = cv2.imencode('.jpg', train_img)[1].tobytes()
            img_str = np.frombuffer(img_str, dtype='uint8')
            #print("Index:{0},name:{1},img_size:{2},str_img_len:{3}, label:{4}".format(count,f_name,train_img.shape,len(img_str),orig_txt))
            if val_idx == val_count_check:
                val_img_list[val_count] = img_str
                val_padded_txt_list[val_count] = train_padded_txt[0] # as this was a list
                val_txt_list[val_count] = orig_txt
                val_label_length_list[val_count] = txt_length
                val_input_length_list[val_count] = max_len
                val_count+=1
                val_idx = 1
            else :
                train_img_list[train_count] = img_str
                train_padded_txt_list[train_count] = train_padded_txt[0]
                train_txt_list[train_count] = orig_txt
                train_label_length_list[train_count] = txt_length
                train_input_length_list[train_count] = max_len
                train_count+=1
                val_idx+=1
                
            
            
            count+=1
            if count == MAX_COUNT:
                flag = 1
                break
            
        if flag == 1:
            break
            
    train_group.attrs['dataset_size'] =  train_count
    val_group.attrs['dataset_size'] =  val_count
    print("Total: {0}, Train :{1}, Test:{2}".format(count,train_count,val_count))
    print('Closing HDF5 file.')
    h5file.close()
    print('Done.')
 


def read_data_file(file_path) :
    f = h5py.File(file_path,'r+')
    train_img_list = list(f['/train/train_img_list'])
    train_txt_list = list(f['/train/train_txt_list'])
    
    length = len(train_img_list)
    print(str(length)+" images in dataset")
    for i in range(length):
        x = train_img_list[i]
        print(train_txt_list[i])
       
        img = cv2.imdecode(x, cv2.IMREAD_COLOR)
        plt.imshow(img)
        plt.show()
        
        
    
    
    print('Closing HDF5 file.')
    f.close()
    print('Done.')

 
def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            #print(char)
            None
        
    return dig_lst 
    
    
def process_data(path,f_name):
 
    print(f_name)
    # read input image and convert into gray scale image
    img = cv2.imread(path)
    if img is None :
        #print(" img not loaded correctly")
        return None,None,None,f_name
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   

    # convert each image of shape (32, 128, 1)
    w,h = img.shape
    #print(img.shape)
    if h > 128 :
        h = 128
    if w > 32:
         w = 32
    
    img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)
    #print(img.shape)
    if w < 32:
        add_zeros = np.ones((32-w, h))*255
        img = np.concatenate((img, add_zeros))

    if h < 128:
        add_zeros = np.ones((32, 128-h))*255
        img = np.concatenate((img, add_zeros), axis=1)
        
    img = np.expand_dims(img , axis = 2)

    # Normalize each image
    #img = img/255.

    # get the text from the image
    txt = f_name.split('_')[1]

    # get maximum length of the text
    txt = txt[:max_label_len]
    training_txt = encode_to_labels(txt)
    train_padded_txt = pad_sequences([training_txt], maxlen=max_label_len, padding='post', value = len(char_list))
    return (img,train_padded_txt,len(txt),txt)


if __name__ == '__main__':
    _main(parser.parse_args())