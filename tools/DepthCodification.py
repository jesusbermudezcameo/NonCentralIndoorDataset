import numpy as np
import cv2

'''
Functions to encode and decode the depth maps from the Non-central Indoor Dataset
As default, the color-coded depth maps from the dataset are coded with a máximum 
    depth of 10.0 meters (default value in the functions)

'decode' function provides a depth maps of the same size of the input image with
    depth values in meters. The input image must be in RGB format with values [0,255]

'encode' function provides a color codification from a depth map in meter values. The
    output will be a color-coded depth map of the same size as the input with a maximum
    value of depth = 'd_max' (set as default to 10.0 meters). The output is in BGR format,
    default image format in openCV library.
'''

def decode(img,d_max=10.):
    d_max = d_max
    R,G,B = img[...,0], img[...,1], img[...,2]
    int1 = d_max/255.0
    int2 = (d_max/255.0)/255.0
    d1 = (R*d_max)/255.0
    d2 = (G/255.0)*int1
    d3 = (B/255.0)*int2
    return d1+d2+d3

def encode(depth,d_max=10.):
    d_max = d_max
    lr = d_max/255.0
    lg = lr/255.0
    lb = lg/255.0
    R=np.floor(depth/lr)
    aux = np.maximum(depth-R*lr,0)
    G = np.floor(aux/lg)
    aux1 = np.maximum(aux-G*lg,0)
    B = np.floor(aux1/lb)
    return cv2.merge((B,G,R))
