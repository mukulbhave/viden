from PIL import Image,ImageOps
import os, sys,re
import numpy as np
import glob

path = "C:\\dataset\\test\\"
out="C:\\dataset\\test\\out\\"

dirs = os.listdir( path )




def resize():
    for index,item in enumerate(dirs):
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = resize_image(im,416,416)
            imResize.save(out + str(index)+'.jpg', 'JPEG', quality=90)




	
def resize_image(img, W,H,debug=True):
    ''' Expects img to be pil image , resizes if image is bigger than target width and height else padds'''
    w,h = img.size
    if debug:
      print(img.size)
    w = W if (w > W) else w
    h = H if ( h > H) else h
    # shrik the image 
    image = img.resize((w,h),Image.BICUBIC)
    w,h = image.size
   
    if w < W:
      w = W-w
      padding = (w//2, 0, w-(w//2), 0)
      image = ImageOps.expand(image, padding)

    if h < H:
      h = H-h
      padding = (0, h//2, 0, h-(h//2))
      image = ImageOps.expand(image, padding)
    if debug:
      print(image.size)
    return image


resize()