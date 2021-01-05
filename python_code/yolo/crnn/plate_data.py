import os
import xml.etree.ElementTree as ElementTree
import glob

import numpy as np
from PIL import Image

input_path =  "C:\\dataset\\viden_data\\"
out_path= "C:\\dataset\\plates\\"

def get_box_for_image(path):
    """Get object bounding boxe annotation for given image."""

   
   
    with open(path) as in_file:
        xml_tree = ElementTree.parse(in_file)
    root = xml_tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        
        xml_box = obj.find('bndbox')
        bbox = ( int(xml_box.find('xmin').text),
                int(xml_box.find('ymin').text), int(xml_box.find('xmax').text),
                int(xml_box.find('ymax').text))
       
    
    return bbox 


i=0
for pathAndFilename in glob.iglob(input_path+"*.jpg"): 
    im = Image.open(pathAndFilename)
    last = pathAndFilename.lower().rfind('jpg')
    xml_name = pathAndFilename[:last]+"xml"
    #xml_name= pathAndFilename.replace('jpg','xml')
    
    print("jpg :"+pathAndFilename)
    print("xml :"+xml_name)
    box = get_box_for_image(xml_name)
    pred_obj_img = im.crop(box)
    pred_obj_img.save(out_path+str(i)+"_.jpg")
    i +=1



