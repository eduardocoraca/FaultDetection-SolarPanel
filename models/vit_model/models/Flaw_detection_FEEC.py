#Created on Nov 28 
import cv2
import numpy as np
from skimage import util
import base64
import tensorflow as tf
import tensorflow_addons as tfa

#############################Area trinca########################################################################

def erase_noisy_points(edges,sarea): #This function erases the last little dots of noise that an image have, and also match some lines that maybe need some dots in the middle.
    auxs=np.uint8(edges)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(auxs, None, None, None, 8, cv2.CV_32S)    
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= sarea:   #keep
            result[labels == i + 1] = 255
    
    #result[result==255]=1
    return result

def proc(image,f1,f2): #Function that process the raw image. It transform the image to gray scale and apply certain filters. The function uses an adaptative threshold.
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    gray=cv2.divide(image, bg, scale=255)
    #gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (5,5),0)
    thresh=cv2.adaptiveThreshold(np.uint8(gray),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,f1,f2)#(21,2) #(101,2)
    #thresh=cv2.adaptiveThreshold(np.uint8(gray),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,199,5)
    thresh=np.invert(thresh)
    thresh=erase_noisy_points(thresh,75)
    #thresh[thresh==255]=1
    thresh=np.uint8(thresh)
    thresh[thresh==1]=255
    thresh=cv2.ximgproc.thinning(thresh, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)#THINNING_ZHANGSUEN THINNING_GUOHALL
    kernel = np.ones((2, 2), 'uint8')
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    thresh[thresh==255]=1

    return thresh

def preprocessing2(image): # Function that finds the grid (vertical and horizontal lines) of a solar cell. 

    gray=image*255
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (300,1))
    horizontal_mask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,500))
    vertical_mask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Combine masks and remove lines
    table_mask = cv2.bitwise_or(horizontal_mask, vertical_mask)
    
    table_mask[table_mask==255]=1
    #table_mask[table_mask==0]=0
    return table_mask

def ho_lines(img): #Function that matches the incomplete lines due to the binarization of the image.
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 50, minLineLength=1, maxLineGap=2)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (255,0), 2)
    img[img==255]=1
    return img


def completing_lines(data): #Function that guarantees complete vertical and horizontal lines.
    for i in range(len(data)-1):
        data[i][np.mean(data[i])>=0.5]=1
   
    for i in range (data.shape[1]-1):
        data[:,i][np.mean(data[:,i])>=0.5]=1   
    return data

def adding_a_frame(image): # Function that adds a frame to help in then the processing.
    vl=[1,len(image.T)-1]
    valh=1
    for i in range(valh):
        image[i]=1

    for i in range(len(image)-valh,len(image)):
        image[i]=1
   
    
    for i in range(0,vl[0]):
        image[:,i]=1
    
    aux=image.shape[1]

    for i in range(vl[1],aux):
        image[:,i]=1
    return image

def sub_lines(imag,lines): #Funtion that substracts one image from another. 
  
  imag[imag==255]=1
  lines[lines==255]=1

  for l in range(len(imag)):
    for e in range(len(imag[l])):
      if imag[l,e]==lines[l,e]:
        imag[l,e]=0
  return imag

def find_hlin(data): #Function that finds the coordinates of each subsection of a cell. 
    l=[]
    aux=0    
    for i in range(len(data[:,50])):          
      if data[i,50]==1 and aux==0:
        if i!=0:
          l.append(i)
        aux=1
      if (data[i,50]==0) and aux==1:
        l.append(i)
        aux=0
    lpaux=[l[i:i+2] for i in range(0,len(l),2)]
    return lpaux

def find_hlinaux(data): #Function that finds the coordinates of the horizontal lines.
    l=[]
    res=[]
    aux=0    
    for i in range(len(data[:,50])):          
      if data[i,50]==1 and aux==0:
        l.append(i)
        aux=1
      if (data[i,50]==0 or i==(len(data)-1)) and aux==1:
        l.append(i)
        aux=0
    lpaux=[l[i:i+2] for i in range(0,len(l),2)]
    
    for e in lpaux:
      res.append(e[1]-e[0])

    return res

def slice_matrix(data,coords): # Function that splits a solar cell in subsections. 
  l=[]
  for e in coords:
    l.append(data[e[0]:e[1],:])
  return l

def evaluate(data): # Function that evaluate and paint the respective affected area, depending of the subsection. 
  res=[]

  for l in range(len(data)):
    data[l][data[l]==1]=255
    datan=np.zeros_like(data[l])
    kernel = np.ones((5, 5), 'uint8')
    data[l] = cv2.morphologyEx(data[l], cv2.MORPH_CLOSE, kernel)
    data[l][data[l]==255]=1
    
    if l==0:
      if data[l].sum()!=0:
        for e in range(len(data[l][1,:])):
          aux=0   
          if data[l][:,e].sum()!=1:
            for i in range(len(data[l][:,e])-3,0,-1):
              if data[l][i-1,e]==1:
                aux=i        
                for j in range(aux):
                    datan[j,e]=1
                break

    elif l ==len(data)-1:
      if data[l].sum()!=0:
        for e in range(len(data[l][1,:])):
          aux=0   
          if data[l][:,e].sum()!=1:
            for i in range(3,len(data[l][:,e])):
              if data[l][i,e]==1:
                aux=i           
                for j in range(len(data[l][:,e]),aux,-1):
                    datan[j-1,e]=1
                break
    else:
      if data[l].sum()!=0:
        for e in range(len(data[l][1,:])):
          aux=[]
          lb=0
          ub=0   
          if data[l][:,e].sum()!=1:
            for i in range(len(data[l][:,e])-2,0,-1):
              if data[l][i-1,e]==1:
                aux.append(i)        
                lb=np.min(aux)
                ub=np.max(aux)
            if (ub-lb>=1):
              for j in range(lb,ub,1):
                  datan[j,e]=1
    res.append(datan)
  
  return res

def conc_subimgs(simgs,len_lin): #Function that concatenates the subsections of the solar cell. 
  aux=[]
  aux2=[]
  for e in range(len(simgs)):
      if e==len(simgs)-1:
        #print(len_lin[e],len(simgs[e]))
        l=np.zeros([len_lin[e]+2,len(simgs[e][0,:])])
        aux.append(l)
      else:
        l=np.zeros([len_lin[e],len(simgs[e][0,:])])
        aux.append(l)

  for e in range(len(aux)):
      
      aux2.append(aux[e])
      aux2.append(simgs[e])

  res=np.vstack(aux2)

  return res

def function_trinca(image):
  
  tm=proc(image,21,2)
  tr=proc(image,101,2)
  tm1=completing_lines(tm)       
  tm1=adding_a_frame(tm1)
  tm1=ho_lines(tm1)
  tm1=preprocessing2(tm1)
  tm1=completing_lines(tm1) 
  tm2=sub_lines(tr,tm1)
  tm2= erase_noisy_points(tm2,20)
  hl=find_hlin(tm1)
  hla=find_hlinaux(tm1)
  tm3=slice_matrix(tm2,hl)
  tm4=evaluate(tm3)
  tm4=conc_subimgs(tm4,hla)
  tm4=cv2.resize(tm4,(tm2.shape[1],tm2.shape[0]))
  tm4=np.logical_or(tm4,tm2) 
  prop=tm4.mean()*100
  prop=round(prop,2)
  
  return tm4, prop

################################################################################################################

def function_outros(img):
    data=np.copy(img)
    data[data<=np.min(data)+25]=1
    data[data>np.min(data)+25]=0
    prop=data.mean()*100
    prop=round(prop,2)
    
    return data, prop

def attention_rollout_map(image, attention_score_dict, model_type = 0, PATCH_SIZE = 7, image_size = (196, 105)):

    num_cls_tokens = 1


    # Average the attention weights across all heads.
    attn_mat = tf.reduce_mean(attention_score_dict, axis=0)


    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_attn = tf.eye(attn_mat.shape[1])
    aug_attn_mat = (attn_mat + residual_attn)
    aug_attn_mat = aug_attn_mat / tf.reduce_sum(aug_attn_mat, axis=-1)[..., None]
    aug_attn_mat = aug_attn_mat.numpy()

    # Recursively multiply the weight matrices.
    joint_attentions = np.zeros(aug_attn_mat.shape)
    joint_attentions[0] = aug_attn_mat[0]

    for n in range(1, aug_attn_mat.shape[0]):
        joint_attentions[n] = np.matmul(joint_attentions[n - 1],aug_attn_mat[n])

    # Attention from the output token to the input space.
    v = (joint_attentions)[-1]
    #grid_size = int(np.sqrt(aug_attn_mat.shape[-1]))
    grid_size = (int(image_size[0]/PATCH_SIZE),int(image_size[1]/PATCH_SIZE))
    mask = v[0,num_cls_tokens:].reshape(grid_size)
    #mask = v[0,num_cls_tokens:].reshape(28, 15)


    aa = mask.copy()
    mask = mask-mask.min()


    mask = cv2.resize((mask / mask.max()), (image.shape[1], image.shape[0]), interpolation = cv2.INTER_LINEAR)#[..., np.newaxis]

    result = (mask * image).astype("uint8")
    return result,mask, aa

def binarize_image(img):    
    ##The following three lines were added##
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
    bg=cv2.morphologyEx(img, cv2.MORPH_DILATE, se)
    G=cv2.divide(img, bg, scale=255)
    #G=np.uint8(img)
    #G = cv2.cvtColor(G, cv2.COLOR_BGR2GRAY)
    G2=cv2.adaptiveThreshold(np.uint8(G),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,3)
    G2 = util.invert(np.uint8(G2))
    G2=erase_noisy_points(G2,75)
    #kernel = np.ones((5, 5), 'uint8')
    G2 = cv2.morphologyEx(G2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    #G2 = cv2.ximgproc.thinning(G2)
    return G2

def clear_cell(imagem):
    
    #Obtaining binary image and auxiliary binary image with filled bussbars
    img = binarize_image(imagem)
    edges = cv2.ximgproc.thinning(img)
    data = np.uint8(edges/255)
    for i in range(len(data)-1):
        data[i][np.mean(data[i])>=0.4]=1
   
    edgesC = data
    
    # This returns an array of r and theta values
    lines = cv2.HoughLines(edgesC, 1, np.pi/180, 70)
    
    
    #degree limit (lines from 0 to lim and 90-lim to 90 degrees are erased)
    #line angles are normalized to be only from 0 to 90 degrees
    lim = 1
    
    mask = np.zeros_like(imagem)
    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        theta_aux =  abs((theta * (180/np.pi))-90)
        #only draws vertical or horizontal lines
        if theta_aux < lim or theta_aux > 90-(lim/2):
            # Stores the value of cos(theta) in a
            a = np.cos(theta)
            # Stores the value of sin(theta) in b
            b = np.sin(theta)    
            # x0 stores the value rcos(theta)
            x0 = a*r         
            # y0 stores the value rsin(theta)
            y0 = b*r         
            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000*(-b))         
            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000*(a))         
            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000*(-b))         
            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000*(a))         
            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (255) denotes the colour of the line to be drawn.
            # In this case, it is white, in order to draw the inpaint mask.
            cv2.line(mask, (x1, y1), (x2, y2), (255), 7)           
    
    dst = cv2.inpaint(imagem,np.uint8(mask),2,cv2.INPAINT_TELEA)
    return dst

def processing_images(dic_info):
    
    dic_out={}
    images=list(dic_info.values())
    labels=list(dic_info.keys())
    for e in range(len(images)):
        if np.mean(images[e])<=25:
            image=np.uint8(np.zeros_like(images[e]))
            dic_out[labels[e]]=image
        else:
            image = images[e] - np.min(images[e])
            image = image/np.max(image)
            image = np.uint8(image*255)
            image= clear_cell(image)
            image = image - np.min(image)
            image = image/np.max(image)
            image = np.uint8(image*255)
            dic_out[labels[e]]=image
    return dic_out

def decode_dic(dic_info):
    
    dic_out={}
    images=list(dic_info.values())
    labels=list(dic_info.keys())
    
    for e in range(len(images)):
        dic_out[labels[e]]=np.frombuffer(base64.b64decode(images[e]),dtype = np.uint8).reshape(280,150)
    
    return dic_out

def loading_models(models_path):
    
    if 'trinca_classifier' not in locals():
        flaw_classifier =  tf.keras.models.load_model(models_path+'vit_classifier_4way_0', compile=False)
        att0 = tf.keras.models.load_model(models_path+'vit_att0_4way_0', compile=False)
        att1 = tf.keras.models.load_model(models_path+'vit_att1_4way_0', compile=False)
    
    return  flaw_classifier, att0, att1


def ml_predict_bs(or_data,dic_info, classifier, att0, att1, batch_size, use_cuda): 
    
    bz=batch_size
    images=list(dic_info.values())
    labels=list(dic_info.keys())
    images=np.asarray(images)
    images=np.expand_dims(images,-1)
    dic_out={}
    
    if use_cuda:
        dev = "gpu"
    else:
        dev = "cpu"

    with tf.device(dev):
        predu_falha_p = classifier.predict(images, batch_size=bz)
    predu_falha= np.argmax(predu_falha_p,axis=1)
    
    sfi=[]
    for i in range(len(predu_falha)):
        if predu_falha[i]==1:
            sfi.append(images[i])
    sfi=np.asarray(sfi)    
   
    img_ar_sf=[]
    area_sf=[] 
    if len(sfi)>0:    
        with tf.device('gpu'):
            hh1 = att1.predict(sfi,batch_size=bz)
            hh0 = att0.predict(sfi,batch_size=bz)
        H = np.concatenate((hh0,hh1),axis = 1)
        
        for e in range(len(H)):
            hd=np.expand_dims(H[e], 0)
            _, attention, _ = attention_rollout_map(sfi[e,:,:,0],hd)
            SF_img = np.where(attention > 2*np.mean(attention),1,0)
            AREA = np.sum(SF_img)/(SF_img.shape[1] * SF_img.shape[0])*100
            AREA=round(AREA,2)
            img_ar_sf.append(SF_img)
            area_sf.append(AREA)
        
    cont=0
    for e in range(len(predu_falha)):
        conc=[]
        if predu_falha[e]==0:
            conc.append('000')
            conc.append(np.uint8(or_data[labels[e]]))
            conc.append(round(np.float64(0),2))
            conc.append(predu_falha_p[e])
            dic_out[labels[e]]=conc
        
        elif predu_falha[e]==1:
            conc.append('010')
            conc.append(np.uint8(255*img_ar_sf[cont]))
            conc.append(area_sf[cont])
            conc.append(predu_falha_p[e])
            dic_out[labels[e]]=conc
            cont=cont+1
        
        elif predu_falha[e]==2:
            conc.append('100')
            
            if np.mean(images[e])<=25:
                result=or_data[labels[e]]
                porc=round(np.float64(100),2)
            else:
                result, porc=function_trinca(or_data[labels[e]])
                result=result*255
            
            conc.append(np.uint8(result))
            conc.append(porc)
            conc.append(predu_falha_p[e])
            dic_out[labels[e]]=conc
        
        else:
            conc.append('001')
            if np.mean(images[e])<=25:
                result=np.ones_like(or_data[labels[e]])
                porc=round(np.float64(100),2)
            else:
                result, porc=function_trinca(or_data[labels[e]])

            conc.append(np.uint8(255*result))
            conc.append(porc)
            conc.append(predu_falha_p[e])
            dic_out[labels[e]]=conc
        
    return dic_out

def encode_response(dic_info):
    data_dic={}
    for e in dic_info.keys():
        data_dic[e]=base64.b64encode(dic_info[e][1]).decode('utf-8')
        data_dic[e+'_label_0']=dic_info[e][0]
        data_dic[e+'_tamanho_0']=dic_info[e][2]
    return data_dic
        

def input_output(encoded_data):
    
    ############### Loading the models  ###################
    models_path='D:\\UNICAMP\\Pro-BYD\\'## Dir of models. Change this Dir when you run this code
    m1,m2,m3=loading_models(models_path)
    #Note: This lines and the function loading_models() can be moved in the main code.
    #######################################################
    
    ################ Decoding the images ##################
    d_data=decode_dic(encoded_data) 
    #Input type = Dic #
    #Output type = Dic #
    #######################################################
    
    ############### Remove the bussbars ###################
    res=processing_images(d_data)
    #Input type = Dic #
    #Output type = Dic #
    #######################################################
    
    ############## Do the predictions #####################
    dic_out = ml_predict_bs(d_data,res, m1, m2, m3)
    #input type(d_data) Dic
    #input type(res) Dic
    #input type(m1,m2,m3) models
    #output type(dic_out) -> Dic
    #######################################################
    
    ########### Enconde the results #######################
    enc_res=encode_response(dic_out)
    #input type(encoded_data) Dic
    #input type(preds) list
    #input type(aff_ar) list
    #input type(out_imgs) list
    #output type(enc_res) Dic
    #######################################################
    return enc_res

if __name__ == '__main__': #### <-This line must be deleted, is here just to separate from the other functions            ###
    
    ### This version uses only on network to classify in 4 classes. The old versions uses 2 networks, the first one to    ###
    ### detect and the second one to classify                                                                             ###
    ### The following line must be in the def(predict) function as showed in the pdf                                      ###
    ### The expected input, encoded_data, is an dictionary containing the labels (1A,1B) and the encoded (base64) images. ### 
    en_res=input_output(encoded_data)
    
    ### The output follows the respective codes                                                                           ###
    ### Codes: Trinca = "100" Solda fria ="010" Celula Ok= "000"                                                          ###
    ### The codes are the same as the presented in the pdf                                                                ###
    
    ### The changes in ml_predict are done. All the model.predict lines are executed using a batch size and the command   ###
    ### 'with tf.device('gpu'):'. All the function returns are dictionaries, with execept of the ml_predict that also     ###
    ### returns other lists containing the attention maps, the affected area and the predictions.                         ###
    ### The return in the function encode_response were also added                                                        ###