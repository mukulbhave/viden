from PIL import Image
import os, sys,re , fnmatch
import numpy as np
import glob

input_path = "C:\\Users\\sudhir\\Downloads\\EngImg\\"
out="C:\\dataset\\viden_numberplates\\out\\"

def rename():
    for index,item in enumerate(dirs):
        
        if  item.endswith(".xml"):
            x=item.find('-') 
            print("Renaming "+item+"  as  "+out +item[x+1:])
        #if os.path.isfile(path+item):
            os.rename(path+item,out +item[x+1:])
            


def convert_png_to_jpg():
    count = 1
    for root, dirnames, filenames in os.walk(input_path):
        print("processing:  "+root)
        for f_name in fnmatch.filter(filenames, '*.png'):
            file_path=os.path.join(root, f_name)
            print("reading file:  "+file_path)
            im = Image.open(file_path)
            rgb_im = im.convert('RGB')
            f_name=f_name.replace(".png",".jpg")
            out_path= os.path.join(out, f_name)
            print("saving:  "+out_path)
            rgb_im.save(out_path, 'JPEG', quality=90)
            
            count+=1
    print("Processed Files:"+str(count))        
            
convert_png_to_jpg()