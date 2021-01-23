import numpy as np
import glob
import os
import cv2 as cv2
from matplotlib import cm
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
from skimage import io ,transform ,feature,measure,filters,exposure
from skimage.color import rgb2gray
from skimage.morphology import binary_erosion, binary_dilation,binary_opening, binary_closing,skeletonize, thin,area_closing,disk
from skimage.util import img_as_ubyte
from collections import Counter 
import csv

from skimage.feature import local_binary_pattern

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val = 0
    x = get_pixel(img, center, x-1, y+1)*1
    val +=x
    val_ar.append(x)     # top_right
    x = get_pixel(img, center, x, y+1)*2   # right
    val +=x
    val_ar.append(x)
    x= get_pixel(img, center, x+1, y+1)*4  # bottom_right
    val +=x
    val_ar.append(x)
    x = get_pixel(img, center, x+1, y)*8    # bottom
    val +=x
    val_ar.append(x)
    x = get_pixel(img, center, x+1, y-1)*16  # bottom_left
    val +=x
    val_ar.append(x)
    x = get_pixel(img, center, x, y-1)*32    # left
    val +=x
    val_ar.append(x)
    x = get_pixel(img, center, x-1, y-1)*64  # top_left
    val +=x
    val_ar.append(x)
    x = get_pixel(img, center, x-1, y)*128    # top
    val +=x
    val_ar.append(x)
    

    return val  


def LBP(img, points_number=8, radius=1, method='default'):

    patch_height = patch_width = 3
    
    lbpImg = np.zeros((img.shape[0],img.shape[1]))
    
    for y in range(0,img.shape[0]-patch_width,patch_width):
        for x in range(0,img.shape[1]-patch_height,patch_height):
            patch = img[y:y+patch_width,x:x+patch_height]
            top_right = patch[0,2]*1
            right = patch[1,2]*2
            bottom_right = patch[2,2]*4
            bottom = patch[2,1]*8
            bottom_left = patch[2,0]*16
            left = patch[1,0]*32
            top_left = patch[0,0]*64
            top = patch[0,1]*128
            sumVal= top_right+right+bottom_right+bottom+bottom_left+left+top_left+top
            lbpImg[y+patch_width//2,x+patch_height//2]=sumVal
            
    lbpImg = lbpImg.astype(int) 
    hist = np.histogram(lbpImg, bins=np.arange(16))
    hist = ((hist[0] - hist[0].min()) / (hist[0].max() - hist[0].min()), hist[1])
    return hist[0]

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 
    
#-------------------------------------------------------BINARIZATION-------------------------------------------------------#

def to_binary(img,val):
    threshold = np.copy(img)
    threshold[threshold<val]=0
    threshold[threshold>=val]=1
    return threshold

#--------------------------------------------------------------------------------------------------------------------------#

def binarize(img):
    
    threshold = filters.threshold_otsu(img)
    img = to_binary(img,threshold)
    return img

#--------------------------------------------------------------------------------------------------------------------------#

def crop_image(img):
    crop_top = 850
    crop_bottom = 400
    crop_right = 15
    crop_left = 150
    
    row = img.shape[0]
    col = img.shape[1]
    
    img = img[crop_top:(row-crop_bottom),crop_left:(col-crop_right)]
    return img


def Run_Length_Encoding(array):
    ones = 0 
    zeros = 0
    output =[]
    output_BW =[]
    for i in range(len(array)):
        
        if array[i] == 0:
            zeros+=1
        elif array[i] == 1:
            ones+=1

        if i+1 < len(array) :
            if array[i+1] != array[i]:
                if array[i] == 0:
                    output.append(zeros)
                    output_BW.append(0)
                    zeros=0
                elif array[i] == 1:
                    output.append(ones)
                    output_BW.append(1)
                    ones=0
                    
        if i+1 == len(array):
            if array[i] == 0:
                output.append(zeros)
                output_BW.append(0)
                zeros=0
            elif array[i] == 1:
                output.append(ones)
                output_BW.append(1)
                ones=0           
                
    return output,output_BW

#--------------------------------------------------------------------------------------------------------------------------#
def most_frequent(lst):
    x= Counter(lst).most_common(1)
    zz = [list(elem) for elem in x]
    for i in range(len(zz)):
        if i == 0:
            if zz[i][1] == 1:
                return min(lst)
        return zz[i][0]
    
#--------------------------------------------------------------------------------------------------------------------------#    
def calculate_reference_lengths(img):
    
    staffspaceheight_arr = []
    stafflineheight_arr = []
    for i in range(img.shape[1]):
        col=img[:,i]
        output,output_BW = Run_Length_Encoding(col)
        ones= []
        zeros = []
        for i in range(len(output)):
            if output_BW[i] ==1:
                ones.append(output[i])
            else:
                zeros.append(output[i])
        staffspaceheight_arr.append(most_frequent(ones))
        stafflineheight_arr.append(most_frequent(zeros))
        
    staffspaceheight = most_frequent(staffspaceheight_arr)
    stafflineheight = most_frequent(stafflineheight_arr )

    return  staffspaceheight , stafflineheight  


def segmentation(img,cust_area,seg,original):
    
    #apply closing 
    closedImage=img
    
    #label the image
    label_image, labelNum = measure.label(closedImage,connectivity=2,background=0,return_num=True)

    
    segmented_notes=[] #list of the segmented notes
    titles=[] #list of strings for image show titles
    i=1 #number of segment for image show titles


    

    
    min_r=[]
    max_r=[]
    
    for region in measure.regionprops(label_image):

        
        # take regions with large enough areas
        if region.area >= cust_area:
            
            
            
            # get rectangle around segmented notes
            minr, minc, maxr, maxc = region.bbox
            
            if (seg == "block")and(maxc-minc >0.7*img.shape[1]):
               
                min_r.append(minr)
                max_r.append(maxr)
                img11 = original[minr:maxr+2,minc:maxc+2]
                
                #add it to the list
                segmented_notes.append(img11)
                titles.append(str(i))
                i+=1
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=1)
                
            if seg == "lines":
                
                tempimg=img[minr:maxr+2,minc:maxc+2]
                
                w=len(tempimg[tempimg==1])
                b=len(tempimg[tempimg==0])
                

                if(b/w > 0.3):
                  
                    img11 = original[minr:maxr+2,minc:maxc+2]

                    min_r.append(minr)
                    max_r.append(maxr)
                    segmented_notes.append(img11)
                    #add it to the list

                    titles.append(str(i))
                    i+=1
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=1)


         
    return segmented_notes,min_r,max_r


def calculateDistance(x1, x2):
    
    distance =np.linalg.norm(x1-x2)
    return distance


def KNN(test_point, training_features, labels, k):
   
    distarr=[]
    for i in range (training_features.shape[0]):
        f=calculateDistance(training_features[i,:],test_point)
        distarr.append(f)
        
        
    sortedarr=np.sort(distarr)
    
    
    classes=np.zeros(3)
    
    for i in range(k):
        result = np.where(distarr == sortedarr[i])
        for j in range(3):
            if(labels[result][0]==j+1):
                classes[j]+=1
            
            
    classification=np.argmax(classes)

  
    return classification+1


def train(direc):
    feature_vector_all=[]
    labels=[]
    direcs = glob.glob (direc)
    min_lines=100
    for direc in direcs:
        files = glob.glob (direc+'/*')
        for file in files:
            #reading an imag
#             print("-----------------------",file,"------------------------------")
            img=io.imread(file)
            #converting it to a gray image
            gray_scale_img=rgb2gray(img)
            max_value=np.max(gray_scale_img)
            min_value=np.min(gray_scale_img)

            if(max_value==1):
                gray_scale_img=(gray_scale_img*255).astype(np.uint8)
            else:
                gray_scale_img=gray_scale_img.astype(np.uint8)




            #binarization process
            binarized_image = binarize(gray_scale_img)

            SE = np.ones((1,3))
            dil = binary_dilation(1-binarized_image,SE)

            segmented_block,min_r,max_r = segmentation(dil,50,"block",binarized_image)
            dil = 1-dil
            cropped_image = dil[max_r[1]:min_r[2],:]

            cropped_image = filters.median(1-cropped_image)


            


            SE = np.ones((1,181))
            dil = binary_dilation(cropped_image,SE)

            segmented_lines,min_r2,max_r2= segmentation(dil,8000,"lines",cropped_image)

        
              
            for i in range(len(segmented_lines)):
                LL = LBP(segmented_lines[i])
                labels.append(int(direc.split('\\')[2]))
                feature_vector_all.append(LL)
                


    return feature_vector_all,labels


def SVM1(feature_vector_all,LL,labels) :
    
    X = np.asarray(feature_vector_all)
    Y = np.asarray(labels)
    X_Test =np.asarray(LL)
    clf = SVC(gamma='scale')
    clf.fit(X, Y)
    new = np.reshape(X_Test, (-1, len(LL)))
    predictions = clf.predict(new)
    return predictions



def KNN1(feature_vector_all,LL,labels,k):
    
    X = np.asarray(feature_vector_all)

    X_Test =np.asarray(LL)
    Y = np.asarray(labels)

   
    knn_prediction = KNN(X_Test,X,Y,k)
    return knn_prediction
    

def preprocessing(img):
    
    gray_scale_img1= (rgb2gray(img)).astype(np.uint8)

    if(np.max(gray_scale_img1)==1):
        gray_scale_img1= (rgb2gray(img)*255).astype(np.uint8)       


    binarized_image = binarize(gray_scale_img1)

   

    return binarized_image
    
def Block_segmentation(img):
    
    binarized_image=img
    SE = np.ones((1,3))
    dil = binary_dilation(1-binarized_image,SE)
    segmented_block,min_r,max_r = segmentation(dil,50,"block",binarized_image)

    dil = 1-dil
    cropped_image = dil[max_r[1]:min_r[2],:]

    return cropped_image


def line_segmentation(img):
    cropped_image=img
    
    cropped_image = filters.median(1-cropped_image)

    SE = np.ones((1,181))
    dil = binary_dilation(cropped_image,SE)

    segmented_lines,min_r2,max_r2= segmentation(dil,8000,"lines",cropped_image)
    
    return segmented_lines,min_r2,max_r2


def feature_extraction(img):
    
    return LBP(img)