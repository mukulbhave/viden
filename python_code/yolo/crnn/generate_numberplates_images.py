from PIL import Image, ImageDraw, ImageFont
import random
import string
import pandas as pd
import re


font_list = [ #ImageFont.truetype('GILSANUB.TTF', 8),ImageFont.truetype('BAUHS93.TTF', 8),
                # ImageFont.truetype('AGENCYR.TTF', 8),ImageFont.truetype('BOD_BLAR.TTF', 8),
                # ImageFont.truetype('BAUHS93.TTF', 8),ImageFont.truetype('BRLNSDB.TTF', 8),                
                # ImageFont.truetype('COLONNA.TTF',8),ImageFont.truetype('ARLRDBD.TTF', 8),
                # ImageFont.truetype('ariblk.ttf', 8),ImageFont.truetype('ALGER.TTF', 8),
                # ImageFont.truetype('calibrib.ttf', 8),
                
                ImageFont.truetype('GILSANUB.TTF', 18),ImageFont.truetype('BAUHS93.TTF', 18),
                ImageFont.truetype('AGENCYR.TTF', 18),ImageFont.truetype('BOD_BLAR.TTF', 18),
                ImageFont.truetype('BAUHS93.TTF', 18),ImageFont.truetype('BRLNSDB.TTF', 18),                
                ImageFont.truetype('COLONNA.TTF',18),ImageFont.truetype('ARLRDBD.TTF', 18),
                ImageFont.truetype('ariblk.ttf', 18),ImageFont.truetype('ALGER.TTF', 18),
                ImageFont.truetype('calibrib.ttf', 18)
                ]
size_for_font = [(15,28),(10,28),(8,28),(12,28),(11,28),(12,28),(11,28),(12,28),(13,28),(12,28),(12,28)]

pattern = re.compile('[\W_]+')

def gen_img(idx,result_str,fnt,min_width,min_height,tilt=0):
    w = len(result_str)*min_width
    h =  random.randint(0,5) + min_height
    img = Image. new('RGB', (w, h), color = (0,0,128))
    d = ImageDraw. Draw(img)
    d. text((5,5), result_str,font=fnt, fill=(255,255,0))
    img=img.rotate(tilt, expand=1)
    result_str=result_str.replace('.','').replace('-','')
    img. save('C:\\git\\yolo\\crnn\\generated_plates\\'+idx+'_'+result_str+".jpg")

#15k only digits , 15k chars and 15k alphnumeric



# def trim_line(line_str):
    # max_len = 31
    # words = line_str.split()
    # temp =''
    # for word in words:
        # word = ''.join(ch for ch in word if ch.isalnum())
        
        # if len(temp+' '+word) > max_len : 
            # break
        # if temp == '' :
            # temp+=word
        # else:
            # temp +=' '+word
    # return temp
        

# books = open('C:\\git\\yolo\\crnn\\books.txt', 'r',encoding="mbcs") 
# lines = books.readlines()

# idx = 1
# for line in lines:
    # line = line.replace('\n','')
    # line = trim_line(line)
    # fnt = idx % 6
    # gen_img(str(idx),line,font_list[fnt],13)
    # idx +=1
    # if idx == 200000:
        # break



idx=0
for i in range(45000):
    tilt=0
    sep = '.' if i%2 == 0 else '-'
    result_str = ''.join(random.choice(string.ascii_uppercase) for j in range(0,2))+sep+''.join(random.choice(string.digits) for j in range(0,2))+''.join(random.choice(string.ascii_uppercase) for j in range(0,2))+sep+''.join(random.choice(string.digits) for j in range(0,4))
    fnt = idx % 11
    w,h = size_for_font[fnt]#10,28
    if i > 15000 and i < 30000: tilt = -10
    elif i >= 30000 : tilt=10
    # elif fnt > 11 : w,h = 30,60
    gen_img(str(idx),result_str,font_list[fnt],w,h,tilt)
    idx +=1
    if idx == 45000:
        break




