from crnn_data_gen import DataGenerator
import h5py
f = h5py.File('C:\\dataset\\CRNN\\train_90kDICT32.hdf5','r+')
train_dataset = f['train']
val_dataset = f['val']

training_generator = DataGenerator(train_dataset, 20,batch_size=15)
validation_generator = DataGenerator(val_dataset, 6,batch_size=5,is_train=0)

print(training_generator.__len__())
print(validation_generator.__len__())
 
for i in range(13):
    training_generator.__getitem__(i)
    print('loop :'+str(i))
    
for i in range (2):
    validation_generator.__getitem__(i)
    print('loop :'+str(i))