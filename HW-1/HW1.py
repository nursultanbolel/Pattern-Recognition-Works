# -*- coding: utf-8 -*-
"""
@name-surname: Nur Sultan BOLEL
@number: 152120151022
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import imageio
import cv2

def showImage(A):
#''' 
#    This function is used to show an image on Concole screen in Spyder.
#        Args:
#        A: is an image which will shown on Concole screen in Spyder.
#'''
    imgplot = plt.imshow(A)
    plt.show()
    
def imageHalf(A):
#''' 
#    This function is used to reduce size of an image by 0.5
#        Args:
#        A: is an image which will resized.
#    
#        Returns:
#        Resized image A
#'''
    A = np.float32(A)
    h,w,d=A.shape
    h=int(h/2)
    w=int(w/2)
    Half_Image = np.arange(h*w*d).reshape(h, w, d)

    for i in range(h):
        for j in range(w):
            Half_Image[i,j,:]=(A[2*i,2*j]+A[(2*i)+1,2*j]+A[2*i,(2*j)+1]+A[(2*i)+1,(2*j)+1])/4
            
    return Half_Image


def createOutput(A,tl,tr,bl,br):
#''' 
#    This function is used to concat of 4 images. 
#    
#        Args:
#        A: it is output image which will include 4 image
#        tl: top left image
#        tr: top right image
#        bl: bottom left
#        br: bottom right
#    
#        Returns:
#        concatted image  
#'''
    h,w,d = A.shape
    border1 = int(h/2)
    border2 = int(w/2)
    A[0:border1,0:border2,:] = tl[:,:,:]
    A[0:border1,border2:w,:] = tr[:,:,:]
    A[border1:h,0:border2,:] = bl[:,:,:]
    A[border1:h,border2:w,:] = br[:,:,:]
    return A

#''' 
#    Main function which tests the functions.
#'''
if __name__ == '__main__':
    
    #cat1.jpg - cat2.jpg
    orginal_Image = imageio.imread('cat2.jpg')
    h,w,d = orginal_Image.shape
    output_Image = np.arange(h*w*d).reshape(h, w, d)
    
    print ('#################\n# ORGINAL IMAGE #\n#################')
    showImage(orginal_Image)
    print("Shape of Image:")
    print(orginal_Image.shape)
    
    print ('#######################\n# HALF RATIO OF IMAGE #\n#######################')
    i_Half = imageHalf(orginal_Image)
    showImage(i_Half)
    print("Shape of Image:")
    print(i_Half.shape)

    h,w,d = i_Half.shape
    i_Green = np.arange(h*w*d).reshape(h, w, d)
    i_Value = np.arange(h*w*d).reshape(h, w, d)
    i_Hue = np.arange(h*w*d).reshape(h, w, d)
    
    print ('########################\n# GREEN SCALE OF IMAGE #\n########################')
    i_Green = i_Half.copy()
    i_Green[:,:,1] = 255
    showImage(i_Green)
    print("Shape of Image:")
    print(i_Green.shape)

    print ('########################\n# VALUE SCALE OF IMAGE #\n########################')           
    i_HSV = cv2.cvtColor(orginal_Image,cv2.COLOR_RGB2HSV)
    i_HSV = imageHalf(i_HSV)
    i_HSV[:,:,2] = 255
    showImage(i_HSV)
    print("Shape of Image:")
    print(i_HSV.shape)
    
    orginal_Image = orginal_Image.astype(np.uint8)
    print ('##########################\n# HUE SCALE OF IMAGE #\n##########################')
    i_HLS = cv2.cvtColor(orginal_Image,cv2.COLOR_RGB2HLS)
    i_HLS = imageHalf(i_HLS)
    i_HLS[:,:,0] = 255
    showImage(i_HLS)
    print("Shape of Image:")
    print(i_HLS.shape)
    
    print ('################\n# OUTPUT IMAGE #\n################')
    output_Image = createOutput(output_Image,i_Half,i_Green,i_HSV,i_HLS)
    output_Image = output_Image.astype(np.uint8)
    showImage(output_Image)
    print("Shape of Image:")
    print(output_Image.shape)
    
    imageio.imwrite('savedCatImage.jpg', output_Image)