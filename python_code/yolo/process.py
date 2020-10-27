import glob, os
from pathlib import Path
import argparse


argparser = argparse.ArgumentParser(
    description="create train.tx and test.txt files.")
argparser.add_argument(
    '-d',
    '--data_path',
    
    help="path to image data file (.jpg files) with trailing \ or /")
    
def _main(args):
# Current directory
    img_path = args.data_path
    print("processing path:"+img_path)
# Directory where the data will reside


# Percentage of images to be used for the test set
    percentage_test = .01;

# Create and/or truncate train.txt and test.txt
    file_train = open(img_path+'train.txt', 'w')  
    file_test = open(img_path+'test.txt', 'w')

# Populate train.txt and test.txt
    counter = 1  
    index_test = round(100 / percentage_test)  
    for pathAndFilename in glob.iglob(img_path+"*.jpg"):  
       title, ext = os.path.splitext(os.path.basename(pathAndFilename))

       if counter == index_test:
           counter = 1
           file_test.write(title +  "\n")
       else:
           file_train.write(title + "\n")
           counter = counter + 1
        

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)       
