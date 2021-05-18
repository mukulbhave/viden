from PIL import Image, ImageDraw, ImageFont
import random
import string
import pandas as pd
import re


# font_list = [   
                # ImageFont.truetype('GILSANUB.TTF', 12),ImageFont.truetype('BAUHS93.TTF', 12),
                # ImageFont.truetype('AGENCYR.TTF', 18),ImageFont.truetype('BOD_BLAR.TTF', 12),
                # ImageFont.truetype('BAUHS93.TTF', 12),ImageFont.truetype('BRLNSDB.TTF', 12),                
                # ImageFont.truetype('COLONNA.TTF',18),ImageFont.truetype('ARLRDBD.TTF', 12),
                # ImageFont.truetype('ariblk.ttf', 12),ImageFont.truetype('ALGER.TTF', 12),
                # ImageFont.truetype('calibrib.ttf', 12),ImageFont.truetype('BOOKOSI.TTF', 12),
                # ImageFont.truetype('BERNHC.TTF', 15),ImageFont.truetype('calibrili.ttf', 12),
                # ImageFont.truetype('BROADW.TTF', 14), ImageFont.truetype('COLONNA.TTF', 15),
                # ImageFont.truetype('COOPBL.TTF', 12), ImageFont.truetype('CASTELAR.TTF', 12),
                # ImageFont.truetype('COPRGTL.TTF', 12),ImageFont.truetype('ELEPHNTI.TTF', 12),
                # ImageFont.truetype('GILLUBCD.TTF', 12),ImageFont.truetype('HATTEN.TTF', 12),
                # ImageFont.truetype('LATINWD.TTF', 12),ImageFont.truetype('ariali.ttf', 12),
                # ImageFont.truetype('IMPRISHA.TTF', 12)
            # ]
font_list = [   
                 ImageFont.truetype('GILSANUB.TTF', 48),ImageFont.truetype('BAUHS93.TTF', 48),
                 ImageFont.truetype('AGENCYR.TTF', 48),ImageFont.truetype('BOD_BLAR.TTF', 48),
                 ImageFont.truetype('BAUHS93.TTF', 48),ImageFont.truetype('BRLNSDB.TTF', 48),                
                 ImageFont.truetype('COLONNA.TTF', 48),ImageFont.truetype('ARLRDBD.TTF', 48),
                 ImageFont.truetype('ariblk.ttf', 48),ImageFont.truetype('ALGER.TTF', 48),
                 ImageFont.truetype('calibrib.ttf', 48),ImageFont.truetype('BOOKOSI.TTF', 48),
                 ImageFont.truetype('BERNHC.TTF', 48),ImageFont.truetype('calibrili.ttf', 48),
                 ImageFont.truetype('BROADW.TTF', 48), ImageFont.truetype('COLONNA.TTF', 48),
                 ImageFont.truetype('COOPBL.TTF', 48), ImageFont.truetype('CASTELAR.TTF', 48),
                 ImageFont.truetype('COPRGTL.TTF', 48),ImageFont.truetype('ELEPHNTI.TTF', 48),
                 ImageFont.truetype('GILLUBCD.TTF', 48),ImageFont.truetype('HATTEN.TTF', 48),
                 ImageFont.truetype('LATINWD.TTF', 35),ImageFont.truetype('ariali.ttf', 48),
                 ImageFont.truetype('IMPRISHA.TTF', 48)
             ]
size_for_font = [
                    (10,28),(10,28),
                    (10,28),(10,28),
                    (10,28),(10,28),
                    (12,28),(10,28),
                    (10,28),(10,28),
                    (10,28),(10,28),
                    (10,28),(10,28),
                    (11,28),(10,28),
                    (10,28),(10,28),
                    (10,28),(10,28),
                    (10,28),(7,28),
                    (15,28),(10,28),
                    (10,28)

                ]

pattern = re.compile('[\W_]+')

def gen_img(idx,result_str,fnt,min_width,min_height,tilt=0):
    w = len(result_str)*min_width
    h =  random.randint(0,5) + min_height
    img = Image. new('RGB', (w, h), color = (0,0,128))
    d = ImageDraw. Draw(img)
    d. text((5,5), result_str,font=fnt, fill=(255,255,0))
    img=img.rotate(tilt, expand=1)
    result_str=result_str.replace('.','').replace('-','')
    img. save('C:\\dataset\\viden_numberplates\\GP_Big\\B-'+idx+'_'+result_str+".jpg")



idx=56001
for i in range(16000):
    tilt=0
    sep = ' ' #if i > 26000 else ''
    result_str = ''.join(random.choice(string.ascii_uppercase) for j in range(0,2))+sep+''.join(random.choice(string.digits) for j in range(0,2))+''.join(random.choice(string.ascii_uppercase) for j in range(0,2))+sep+''.join(random.choice(string.digits) for j in range(0,4))
    fnt = idx % 25
    w,h = 45,100#size_for_font[fnt]
    # if i > 15000 and i < 30000: tilt = -10
    # elif i >= 30000 : tilt=10
    
    gen_img(str(idx),result_str,font_list[fnt],w,h,tilt)
    idx +=1
    




