import os
import xml.etree.ElementTree as ElementTree
import glob
import ntpath
import numpy as np
from PIL import Image

#input_path =  "C:\\Users\\sudhir\\Downloads\\CCPD2019\\ccpd_base\\"

input_path="C:\\Users\\sudhir\\Downloads\\CCPD2019\\ccpd_rotate\\"
out_path= "C:\\dataset\\plates\\ccpd\\"

alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
def get_box_for_image(path):
    
       xy = path.split("_")
       xymin=xy[0].split("&")
       xymax=xy[1].split("&")
       return (int(xymin[0]),int(xymin[1]),int(xymax[0]),int(xymax[1]))


def get_label(path):
    number = path.split("_")[1:]
    alpha = number[0]
    number = number[1:]
    label= alphabets[int(alpha)]
    for n in number:
        label +=ads[int(n)]
    return label

i=575
for pathAndFilename in glob.iglob(input_path+"*.jpg"): 
    im = Image.open(pathAndFilename)
    last = pathAndFilename.lower().rfind('jpg')
    head,fname = ntpath.split(pathAndFilename)
    fparts = fname.split("-")
    
    box = get_box_for_image(fparts[2])
    label = get_label(fparts[4])
    pred_obj_img = im.crop(box)
    pred_obj_img.save(out_path+str(i)+"_"+label+".jpg")
    i +=1
    if(i == 10000): break



