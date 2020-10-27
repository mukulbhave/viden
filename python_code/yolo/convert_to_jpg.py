from PIL import Image
import os, sys,re
import numpy as np
import glob

path = "C:\git\yolo\img-books\data\\"
out="C:\git\yolo\img-books\data\\"

dirs = os.listdir( path )
def saveAsJpg():
    for index,item in enumerate(dirs):
        print(item)
        if  item.endswith(".png"):
        #if os.path.isfile(path+item):
            im = Image.open(path+item)
            rgb_im = im.convert('RGB')
            f, e = os.path.splitext(path+item)
            rgb_im.save(out + str(index)+'.jpg', 'JPEG', quality=90)
            
def rename():
    for index,item in enumerate(dirs):
        
        if  item.endswith(".xml"):
            x=item.find('-') 
            print("Renaming "+item+"  as  "+out +item[x+1:])
        #if os.path.isfile(path+item):
            os.rename(path+item,out +item[x+1:])
            



rename()