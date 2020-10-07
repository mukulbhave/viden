#Create mat iterates ovethe file name pattern in dir and creates a mat file with name as output_path
def create_mat_file(img_path_pattern,output_path) :
""" A helper function to create  data in mat format  """
    imglist = [f for f in glob.glob(input_path)]
    size= len(imglist)
    dataX = np.empty([size,30,30,3])
    dataY = np.empty([size,1])
    for img in imglist :
        pil_im = Image.open(img).resize((30,30))
        x = np.asarray(pil_im)
        np.append(dataX,x)
        np.append(dataY,img)
    
    data = {}
    data['X'] = dataX
    data['Y'] = dataY
    savemat(output_path,data,True)