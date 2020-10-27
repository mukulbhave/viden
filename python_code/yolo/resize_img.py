from PIL import Image
import os, sys,re
import numpy as np
import glob

path = "D:\\MUKUL\\git\\yolo\img-books\\"
out="D:\\MUKUL\\git\\yolo\img-books\\data\\"

dirs = os.listdir( path )




def resize():
    for index,item in enumerate(dirs):
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((608,608), Image.ANTIALIAS)
            imResize.save(out + str(index)+'.jpg', 'JPEG', quality=90)



def createMatFile(input_path,output_path) :
    """ A helper function to create  data in mat format  """
    # imglist = [f for f in glob.glob(input_path)]
    # size= len(imglist)
    # dataX = np.empty([size,imglist[0].shape)
    # dataY = np.empty([size,1])
    # for img in imglist :
        # pil_im = Image.open(img)
		# name=os.path.splitext(img)[0]
        # x = np.asarray(pil_im)
        # np.append(dataX,x)
        # np.append(dataY,img)
    
    # data = {}
    # data['X'] = dataX
    # data['Y'] = dataY
    # savemat(output_path,data,True)
	



resize()