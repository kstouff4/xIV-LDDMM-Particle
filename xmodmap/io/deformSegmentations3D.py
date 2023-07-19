# Libraries to Import

import numpy as np
import math

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec as gs
import os
import glob
import scipy.interpolate as spi
import scipy as sp
from scipy import stats
import matplotlib.cm as cm
from matplotlib import colors

# Tools for PCA
import glob
# these methods from scipy will be used for displaying some images
from scipy.linalg import eigh
from scipy.stats import norm

from PIL import Image
Image.MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS=1e10 # forget attack

# Tools for Generating Paths
from numpy import linalg as LA
from itertools import combinations
from scipy.special import comb
from scipy import stats

# Tools for caching
from os import path

# Tools for saving images
import scipy.misc
from scipy import io

# Tools for normalizing MRI images
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk 
from matplotlib import pyplot

# Tools for Error Checking
from scipy.spatial.distance import directed_hausdorff
from numpy import random
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Tools for Reading in File
import nibabel as nib
from nibabel import processing
from nibabel import funcs

# Tools for Marching Cubes
import skimage 
from skimage import measure

import pandas

import matplotlib.patches as mpatches
import tifffile as tiff

# Tools for visualization harnessed from 3D-Surface-Generator (only works if no)
#from ipynb.fs.full.SurfaceGenerator3D import draw

import vtkFunctions as vt
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util.numpy_support import vtk_to_numpy

######################################################################
# Interpolation Functions 

# use mode interpolation frequently
# Daniel's method of interpretation 
def interp3(X0, X1, X2, I, X0s, X1s, X2s, bc='nearest'):
    ''' linear interpolation
    I want this to work for I of arbitrary dimension, interpolating across the first 2
    
    Args:
    X0 = spatial coordinates of domain (1 x X)
    X1 = spatial coordinates of domain (1 x Y)
    X2 = spatial coordinates of domain (1 x Z)
    I = function values
    X0s = where we want to find function values (3D grid with x coord)
    X1s = where we want to find function values (3D grid with y coord)
    X2s = where we want to find function values (3D grid with z coord)
    
    returns function value (3 space)
    '''
    # convert sample points to index coords (since from matlab, should be xy only????)
    X0_1d = X0
    X1_1d = X1
    X2_1d = X2
    
    # squeeze if first dimension is 1 and exists second 
    if len(X0.shape) > 1:
        X0_1d= np.squeeze(X0)
    if len(X1.shape) > 1:
        X1_1d = np.squeeze(X1)
    if len(X2.shape) > 1:
        X2_1d = np.squeeze(X2)
        
    # convert sample points to index coords (since from matlab, should be xy only????)
    if (X0_1d.shape[0] < 2):
        dx0 = 1 # don't know how to scale this
        X0si = (X0s - X0_1d[0])/dx0
    else:
        dx0 = X0_1d[1] - X0_1d[0]
        X0si = (X0s - X0_1d[0])/dx0
    if (X1_1d.shape[0] < 2):
        dx1 = 1 # don't know how to scale this
        X1si = (X1s - X1_1d[0])/dx1
    else:
        dx1 = X1_1d[1] - X1_1d[0]
        X1si = (X1s - X1_1d[0])/dx1
    if (X2_1d.shape[0] < 2):
        dx2 = 1 # don't know how to scale this
        X2si = (X2s - X2_1d[0])/dx2
    else:
        dx2 = X2_1d[1] - X2_1d[0]
        X2si = (X2s - X2_1d[0])/dx2
    print("Differences are: " + str(dx0) + ", " + str(dx1) + ", " + str(dx2))
    print("Should be 0.125 mm")
    
    
    # get fraction to next for weights
    X0si0 = np.floor(X0si).astype(int)
    X1si0 = np.floor(X1si).astype(int)
    X2si0 = np.floor(X2si).astype(int)
    p0 = X0si - X0si0
    p1 = X1si - X1si0
    p2 = X2si - X2si0
    X0si1 = X0si0+1
    X1si1 = X1si0+1
    X2si1 = X2si0+1
    
    # add necessary axes to p
    nadd = len(I.shape)-3
    for i in range(nadd):
        p0 = p0[...,None]
        p1 = p1[...,None]
        p2 = p2[...,None]
    
    # boundary conditions.  This is nearest neighbor extrapolation which is usually appropriate
    # 0 denotes floor being out of bounds, 1 denotes ceiling being out of bounds
    if isinstance(bc,np.ndarray):
        #floor for x, y, or z index is out of bounds
        bad000 = X0si0<0
        bad000 = np.logical_or(bad000,X0si0>I.shape[0]-1) # x0 coordinate is out of bounds
        bad000 = np.logical_or(bad000,X1si0<0) # x1 coordinate is out of bounds (small)
        bad000 = np.logical_or(bad000,X1si0>I.shape[1]-1) #x1 coordinate is out of bounds (large)
        bad000 = np.logical_or(bad000,X2si0<0)
        bad000 = np.logical_or(bad000,X2si0<I.shape[2]-1)
        
        # ceiling for x is out of bounds or floor for y,z
        bad100 = X0si1<0 # means 0 is incremented
        bad100 = np.logical_or(bad100,X0si1>I.shape[0]-1)
        bad100 = np.logical_or(bad100,X1si0<0)
        bad100 = np.logical_or(bad100,X1si0>I.shape[1]-1)
        bad100 = np.logical_or(bad100,X2si0<0)
        bad100 = np.logical_or(bad100,X2si0>I.shape[2]-1)
        
        bad010 = X0si0<0 # means 1 is incremented
        bad010 = np.logical_or(bad010,X0si0>I.shape[0]-1)
        bad010 = np.logical_or(bad010,X1si1<0)
        bad010 = np.logical_or(bad010,X1si1>I.shape[1]-1)
        bad010 = np.logical_or(bad010,X2si0<0)
        bad010 = np.logical_or(bad010,X2si0>I.shape[2]-1)
        
        bad110 = X0si1<0 # means 0 and 1 is incremented
        bad110 = np.logical_or(bad110,X0si1>I.shape[0]-1)
        bad110 = np.logical_or(bad110,X1si1<0)
        bad110 = np.logical_or(bad110,X1si1>I.shape[1]-1)
        bad110 = np.logical_or(bad110,X2si0<0)
        bad110 = np.logical_or(bad110,X2si0>I.shape[2]-1)
        
        bad001 = X0si0<0 # means 0 and 1 is incremented
        bad001 = np.logical_or(bad001,X0si0>I.shape[0]-1)
        bad001 = np.logical_or(bad001,X1si0<0)
        bad001 = np.logical_or(bad001,X1si0>I.shape[1]-1)
        bad001 = np.logical_or(bad001,X2si1<0)
        bad001 = np.logical_or(bad001,X2si1>I.shape[2]-1)
        
        bad011 = X0si0<0 # means 0 and 1 is incremented
        bad011 = np.logical_or(bad011,X0si0>I.shape[0]-1)
        bad011 = np.logical_or(bad011,X1si1<0)
        bad011 = np.logical_or(bad011,X1si1>I.shape[1]-1)
        bad011 = np.logical_or(bad011,X2si1<0)
        bad011 = np.logical_or(bad011,X2si1>I.shape[2]-1)

        bad101 = X0si1<0 # means 0 and 1 is incremented
        bad101 = np.logical_or(bad101,X0si1>I.shape[0]-1)
        bad101 = np.logical_or(bad101,X1si0<0)
        bad101 = np.logical_or(bad101,X1si0>I.shape[1]-1)
        bad101 = np.logical_or(bad101,X2si1<0)
        bad101 = np.logical_or(bad101,X2si1>I.shape[2]-1)
        
        bad111 = X0si1<0 # means 0 and 1 is incremented
        bad111 = np.logical_or(bad111,X0si1>I.shape[0]-1)
        bad111 = np.logical_or(bad111,X1si1<0)
        bad111 = np.logical_or(bad111,X1si1>I.shape[1]-1)
        bad111 = np.logical_or(bad111,X2si1<0)
        bad111 = np.logical_or(bad111,X2si1>I.shape[2]-1)
       
            
    # set boundary conditions to nearest        
    X0si0[X0si0<0] = 0
    X0si0[X0si0>I.shape[0]-1] = I.shape[0]-1
    X1si0[X1si0<0] = 0
    X1si0[X1si0>I.shape[1]-1] = I.shape[1]-1
    X2si0[X2si0<0] = 0
    X2si0[X2si0>I.shape[2]-1] = I.shape[2]-1
    X0si1[X0si1<0] = 0
    X0si1[X0si1>I.shape[0]-1] = I.shape[0]-1
    X1si1[X1si1<0] = 0
    X1si1[X1si1>I.shape[1]-1] = I.shape[1]-1
    X2si1[X2si1<0] = 0
    X2si1[X2si1>I.shape[2]-1] = I.shape[2]-1
    
    
    # vectorize (note that ravel and reshape iterate over z, then y, then x)
    # in other words, x is outer loop, y is middle, z is inner loop
    # All the floor and ceiling coordinates in a list
    X0si0 = X0si0.ravel()
    X0si1 = X0si1.ravel()
    X1si0 = X1si0.ravel()
    X1si1 = X1si1.ravel()
    X2si0 = X2si0.ravel()
    X2si1 = X2si1.ravel()
    
    # this is if ravel would go down columns
    #X00 = X0si0 + X1si0*I.shape[0]
    #X01 = X0si0 + X1si1*I.shape[0]
    #X10 = X0si1 + X1si0*I.shape[0]
    #X11 = X0si1 + X1si1*I.shape[0]
    
    # this is if ravel goes across rows
    '''
    X000 = X0si0*I.shape[1]*I.shape[2] + X1si0 + I.shape[1]*X2si0
    X010 = X0si0*I.shape[1]*I.shape[2] + X1si1 + I.shape[1]*X2si0
    X001 = X0si0*I.shape[1]*I.shape[2] + X1si0 + I.shape[1]*X2si1
    X011 = X0si0*I.shape[1]*I.shape[2] + X1si1 + I.shape[1]*X2si1
    X100 = X0si1*I.shape[1]*I.shape[2] + X1si0 + I.shape[1]*X2si0
    X110 = X0si1*I.shape[1]*I.shape[2] + X1si1 + I.shape[1]*X2si0
    X101 = X0si1*I.shape[1]*I.shape[2] + X1si0 + I.shape[1]*X2si1
    X111 = X0si1*I.shape[1]*I.shape[2] + X1si1 + I.shape[1]*X2si1
    '''
    
    dims = (I.shape[0], I.shape[1], I.shape[2])
    
    # previous part redone re: Daniel's suggestion
    '''
    X000 = X0si0*I.shape[1]*I.shape[2] + X1si0*I.shape[2] + X2si0
    X010 = X0si0*I.shape[1]*I.shape[2] + X1si1*I.shape[2] + X2si0
    X001 = X0si0*I.shape[1]*I.shape[2] + X1si0*I.shape[2] + X2si1
    X011 = X0si0*I.shape[1]*I.shape[2] + X1si1*I.shape[2] + X2si1
    X100 = X0si1*I.shape[1]*I.shape[2] + X1si0*I.shape[2] + X2si0
    X110 = X0si1*I.shape[1]*I.shape[2] + X1si1*I.shape[2] + X2si0
    X101 = X0si1*I.shape[1]*I.shape[2] + X1si0*I.shape[2] + X2si1
    X111 = X0si1*I.shape[1]*I.shape[2] + X1si1*I.shape[2] + X2si1
    '''
    
    X000 = np.ravel_multi_index([X0si0, X1si0, X2si0], dims)
    X010 = np.ravel_multi_index([X0si0, X1si1, X2si0], dims)
    X001 = np.ravel_multi_index([X0si0, X1si0, X2si1], dims)
    X011 = np.ravel_multi_index([X0si0, X1si1, X2si1], dims)
    X100 = np.ravel_multi_index([X0si1, X1si0, X2si0], dims)
    X110 = np.ravel_multi_index([X0si1, X1si1, X2si0], dims)
    X101 = np.ravel_multi_index([X0si1, X1si0, X2si1], dims)
    X111 = np.ravel_multi_index([X0si1, X1si1, X2si1], dims)
    
                
    # sample 8 times (all combinations of floor and ceiling)
    # input shape
    nI = list(I.shape)
    nIravel = [nI[0]*nI[1]*nI[2]]
    nIravel.extend(nI[3:]) # extend to accomodate all dimensions (assume that functions are at least of 1 dim)
    
    # output shape
    n = list(X0s.shape)
    n.extend(nI[3:])
    #nravel = [n[0]*n[1]]
    #nravel.extend(n[2:])
    
    I_ = np.reshape(I,nIravel)    
    I000 = np.reshape(I_[X000],n)
    I010 = np.reshape(I_[X010],n)
    I001 = np.reshape(I_[X001],n)
    I100 = np.reshape(I_[X100],n)
    I110 = np.reshape(I_[X110],n)
    I101 = np.reshape(I_[X101],n)
    I011 = np.reshape(I_[X011],n)
    I111 = np.reshape(I_[X111],n)
    
    if isinstance(bc,np.ndarray):      
        # set out of bounds to constant
        I000[bad000] = bc
        I010[bad010] = bc
        I100[bad100] = bc
        I110[bad110] = bc
        I001[bad001] = bc
        I101[bad101] = bc
        I011[bad011] = bc
        I111[bad111] = bc
        
    # output (1 - p for 0s; smaller the p the closer to floor value)
    # weighted average of samples by how close index is to floor vs. ceiling 
    return I000*((1.0-p0)*(1.0-p1)*(1.0-p2)) + \
           I010*((1.0-p0)*(    p1)*(1.0-p2)) + \
           I100*((    p0)*(1.0-p1)*(1.0-p2)) + \
           I110*((    p0)*(    p1)*(1.0-p2)) + \
            I001*((1.0-p0)*(1.0-p1)*(p2)) + \
            I011*((1.0-p0)*(    p1)*(p2)) + \
            I101*((    p0)*(1.0-p1)*(p2)) + \
            I111*((    p0)*(    p1)*(p2))  

def interp3NN(X0, X1, X2, I, X0s, X1s, X2s, bc='nearest'):
    ''' NN interpolation
    I want this to work for I of arbitrary dimension, interpolating across the first 2
    
    Args:
    X0 = spatial coordinates of domain (1 x X)
    X1 = spatial coordinates of domain (1 x Y)
    X2 = spatial coordinates of domain (1 x Z)
    I = function values
    X0s = where we want to find function values (3D grid with x coord)
    X1s = where we want to find function values (3D grid with y coord)
    X2s = where we want to find function values (3D grid with z coord)
    
    returns function value (3 space)
    '''
    X0_1d = np.squeeze(X0)
    X1_1d = np.squeeze(X1)
    X2_1d = np.squeeze(X2)
    # convert sample points to index coords (since from matlab, should be xy only????)
    dx0 = X0_1d[1]- X0_1d[0]
    dx1 = X1_1d[1]- X1_1d[0]
    dx2 = X2_1d[1] - X2_1d[0]
    X0si = (X0s - X0_1d[0])/dx0
    X1si = (X1s - X1_1d[0])/dx1 
    X2si = (X2s - X2_1d[0])/dx2
    
    # debugging
    print("shapes of coords in interp3NN:")
    print(X0si.shape)
    print(X1si.shape)
    print(X2si.shape)
    
    # get fraction to next for weights
    X0si0 = np.floor(X0si).astype(int)
    X1si0 = np.floor(X1si).astype(int)
    X2si0 = np.floor(X2si).astype(int)
    p0 = X0si - X0si0
    p1 = X1si - X1si0
    p2 = X2si - X2si0
    X0si1 = X0si0+1
    X1si1 = X1si0+1
    X2si1 = X2si0+1
    
    # save lengths
    # Make sure you are within boundary
    X0si1[X0si1 >= len(np.squeeze(X0))] = len(np.squeeze(X0)) - 1
    X0si1[X0si1 < 0] = 0
    X0si0[X0si0 >= len(np.squeeze(X0))] = len(np.squeeze(X0)) - 1
    X0si0[X0si0 < 0] = 0
    X1si1[X1si1 >= len(np.squeeze(X1))] = len(np.squeeze(X1)) - 1
    X1si1[X1si1 < 0] = 0
    X1si0[X1si0 >= len(np.squeeze(X1))] = len(np.squeeze(X1)) - 1
    X1si0[X1si0 < 0] = 0
    X2si1[X2si1 >= len(np.squeeze(X2))] = len(np.squeeze(X2)) - 1
    X2si1[X2si1 < 0] = 0
    X2si0[X2si0 >= len(np.squeeze(X2))] = len(np.squeeze(X2)) - 1
    X2si0[X2si0 < 0] = 0
    
    X0si = X0si.ravel()
    X1si = X1si.ravel()
    X2si = X2si.ravel()
    X0si0 = X0si0.ravel()
    X0si1 = X0si1.ravel()
    X1si0 = X1si0.ravel()
    X1si1 = X1si1.ravel()
    X2si0 = X2si0.ravel()
    X2si1 = X2si1.ravel()
    
    X0si = (X0si <= X0si0+0.5)*X0si0 + (X0si > X0si0+0.5)*X0si1
    X1si = (X1si <= X1si0+0.5)*X1si0 + (X1si > X1si0+0.5)*X1si1
    X2si= (X2si <= X2si0+0.5)*X2si0 + (X2si > X2si0+0.5)*X2si1
    
    X0NN = X0si
    X1NN = X1si
    X2NN = X2si


    # this is if ravel goes across rows
    XNN = X0NN*I.shape[1]*I.shape[2] + X1NN*I.shape[2] + X2NN
                
    # input shape
    nI = list(I.shape)
    nIravel = [nI[0]*nI[1]*nI[2]]
    nIravel.extend(nI[3:]) # extend to accomodate all dimensions (assume that functions are at least of 1 dim)
    
    # output shape
    n = list(X0s.shape)
    n.extend(nI[3:])
    
    I_ = np.reshape(I,nIravel)    
    INN = np.reshape(I_[XNN],n)
    
    # no need for boundary conditions??
    
    return INN

def interp3Mode(X0, X1, X2, I, X0s, X1s, X2s, bc='nearest'):
    '''
    Take a weighted mode of 8 NN
    Args:
    
    X0 = spatial coordinates of domain (1 x Y)
    X1 = spatial coordinates of domain (1 x X)
    X2 = spatial coordinates of domain (1 x Z)
    I = function values (assume certain number of unique values for labels)
    X0s = where we want to find function values (3D grid with x coord)
    X1s = where we want to find function values (3D grid with y coord)
    X2s = where we want to find function values (3D grid with z coord)
    
    returns function value (3 space)
    '''
    # convert sample points to index coords (since from matlab, should be xy only????)
    X0_1d = X0.astype('float32')
    X1_1d = X1.astype('float32')
    X2_1d = X2.astype('float32')
    
    # squeeze if first dimension is 1 and exists second 
    if len(X0.shape) > 1:
        X0_1d= np.squeeze(X0.astype('float32'))
    if len(X1.shape) > 1:
        X1_1d = np.squeeze(X1.astype('float32'))
    if len(X2.shape) > 1:
        X2_1d = np.squeeze(X2.astype('float32'))
        
    # convert sample points to index coords (since from matlab, should be xy only????)
    if (X0_1d.shape[0] < 2):
        dx0 = 1 # don't know how to scale this
        X0si = (X0s - X0_1d[0])/dx0
    else:
        dx0 = X0_1d[1] - X0_1d[0]
        X0si = (X0s - X0_1d[0])/dx0
    if (X1_1d.shape[0] < 2):
        dx1 = 1 # don't know how to scale this
        X1si = (X1s - X1_1d[0])/dx1
    else:
        dx1 = X1_1d[1] - X1_1d[0]
        X1si = (X1s - X1_1d[0])/dx1
    if (X2_1d.shape[0] < 2):
        dx2 = 1 # don't know how to scale this
        X2si = (X2s - X2_1d[0])/dx2
    else:
        dx2 = X2_1d[1] - X2_1d[0]
        X2si = (X2s - X2_1d[0])/dx2
    print("Differences are: " + str(dx0) + ", " + str(dx1) + ", " + str(dx2))
    print("Should be 0.125 mm")
    
    
    # get fraction to next for weights
    X0si0 = np.floor(X0si).astype(int)
    X1si0 = np.floor(X1si).astype(int)
    X2si0 = np.floor(X2si).astype(int)
    p0 = (X0si - X0si0).astype('float32')
    p1 = (X1si - X1si0).astype('float32')
    p2 = (X2si - X2si0).astype('float32')
    X0si1 = X0si0+1
    X1si1 = X1si0+1
    X2si1 = X2si0+1
    
    # add necessary axes to p
    nadd = len(I.shape)-3
    for i in range(nadd):
        p0 = p0[...,None]
        p1 = p1[...,None]
        p2 = p2[...,None]
    
    # boundary conditions.  This is nearest neighbor extrapolation which is usually appropriate
    # 0 denotes floor being out of bounds, 1 denotes ceiling being out of bounds
    if isinstance(bc,np.ndarray):
        #floor for x, y, or z index is out of bounds
        bad000 = X0si0<0
        bad000 = np.logical_or(bad000,X0si0>I.shape[0]-1) # x0 coordinate is out of bounds
        bad000 = np.logical_or(bad000,X1si0<0) # x1 coordinate is out of bounds (small)
        bad000 = np.logical_or(bad000,X1si0>I.shape[1]-1) #x1 coordinate is out of bounds (large)
        bad000 = np.logical_or(bad000,X2si0<0)
        bad000 = np.logical_or(bad000,X2si0<I.shape[2]-1)
        
        # ceiling for x is out of bounds or floor for y,z
        bad100 = X0si1<0 # means 0 is incremented
        bad100 = np.logical_or(bad100,X0si1>I.shape[0]-1)
        bad100 = np.logical_or(bad100,X1si0<0)
        bad100 = np.logical_or(bad100,X1si0>I.shape[1]-1)
        bad100 = np.logical_or(bad100,X2si0<0)
        bad100 = np.logical_or(bad100,X2si0>I.shape[2]-1)
        
        bad010 = X0si0<0 # means 1 is incremented
        bad010 = np.logical_or(bad010,X0si0>I.shape[0]-1)
        bad010 = np.logical_or(bad010,X1si1<0)
        bad010 = np.logical_or(bad010,X1si1>I.shape[1]-1)
        bad010 = np.logical_or(bad010,X2si0<0)
        bad010 = np.logical_or(bad010,X2si0>I.shape[2]-1)
        
        bad110 = X0si1<0 # means 0 and 1 is incremented
        bad110 = np.logical_or(bad110,X0si1>I.shape[0]-1)
        bad110 = np.logical_or(bad110,X1si1<0)
        bad110 = np.logical_or(bad110,X1si1>I.shape[1]-1)
        bad110 = np.logical_or(bad110,X2si0<0)
        bad110 = np.logical_or(bad110,X2si0>I.shape[2]-1)
        
        bad001 = X0si0<0 # means 0 and 1 is incremented
        bad001 = np.logical_or(bad001,X0si0>I.shape[0]-1)
        bad001 = np.logical_or(bad001,X1si0<0)
        bad001 = np.logical_or(bad001,X1si0>I.shape[1]-1)
        bad001 = np.logical_or(bad001,X2si1<0)
        bad001 = np.logical_or(bad001,X2si1>I.shape[2]-1)
        
        bad011 = X0si0<0 # means 0 and 1 is incremented
        bad011 = np.logical_or(bad011,X0si0>I.shape[0]-1)
        bad011 = np.logical_or(bad011,X1si1<0)
        bad011 = np.logical_or(bad011,X1si1>I.shape[1]-1)
        bad011 = np.logical_or(bad011,X2si1<0)
        bad011 = np.logical_or(bad011,X2si1>I.shape[2]-1)

        bad101 = X0si1<0 # means 0 and 1 is incremented
        bad101 = np.logical_or(bad101,X0si1>I.shape[0]-1)
        bad101 = np.logical_or(bad101,X1si0<0)
        bad101 = np.logical_or(bad101,X1si0>I.shape[1]-1)
        bad101 = np.logical_or(bad101,X2si1<0)
        bad101 = np.logical_or(bad101,X2si1>I.shape[2]-1)
        
        bad111 = X0si1<0 # means 0 and 1 is incremented
        bad111 = np.logical_or(bad111,X0si1>I.shape[0]-1)
        bad111 = np.logical_or(bad111,X1si1<0)
        bad111 = np.logical_or(bad111,X1si1>I.shape[1]-1)
        bad111 = np.logical_or(bad111,X2si1<0)
        bad111 = np.logical_or(bad111,X2si1>I.shape[2]-1)
       
            
    # set boundary conditions to nearest        
    X0si0[X0si0<0] = 0
    X0si0[X0si0>I.shape[0]-1] = I.shape[0]-1
    X1si0[X1si0<0] = 0
    X1si0[X1si0>I.shape[1]-1] = I.shape[1]-1
    X2si0[X2si0<0] = 0
    X2si0[X2si0>I.shape[2]-1] = I.shape[2]-1
    X0si1[X0si1<0] = 0
    X0si1[X0si1>I.shape[0]-1] = I.shape[0]-1
    X1si1[X1si1<0] = 0
    X1si1[X1si1>I.shape[1]-1] = I.shape[1]-1
    X2si1[X2si1<0] = 0
    X2si1[X2si1>I.shape[2]-1] = I.shape[2]-1
    
    
    # vectorize (note that ravel and reshape iterate over z, then y, then x)
    # in other words, x is outer loop, y is middle, z is inner loop
    # All the floor and ceiling coordinates in a list
    X0si0 = X0si0.ravel()
    X0si1 = X0si1.ravel()
    X1si0 = X1si0.ravel()
    X1si1 = X1si1.ravel()
    X2si0 = X2si0.ravel()
    X2si1 = X2si1.ravel()
    
    print("shapes of p's")
    print(p0.shape)
    print(p1.shape)
    print(p2.shape)
    
    p0 = p0.ravel()
    p1 = p1.ravel()
    p2 = p2.ravel()
    

    
    dims = (I.shape[0], I.shape[1], I.shape[2])
    
    X000 = np.ravel_multi_index([X0si0, X1si0, X2si0], dims)
    X010 = np.ravel_multi_index([X0si0, X1si1, X2si0], dims)
    X001 = np.ravel_multi_index([X0si0, X1si0, X2si1], dims)
    X011 = np.ravel_multi_index([X0si0, X1si1, X2si1], dims)
    X100 = np.ravel_multi_index([X0si1, X1si0, X2si0], dims)
    X110 = np.ravel_multi_index([X0si1, X1si1, X2si0], dims)
    X101 = np.ravel_multi_index([X0si1, X1si0, X2si1], dims)
    X111 = np.ravel_multi_index([X0si1, X1si1, X2si1], dims)

    p000 = ((1.0-p0)*(1.0-p1)*(1.0-p2)).astype('float32')
    p010 = ((1.0-p0)*(p1)*(1.0-p2)).astype('float32')
    p001 = ((1.0-p0)*(1.0-p1)*(p2)).astype('float32')
    p011 = ((1.0-p0)*(p1)*(p2)).astype('float32')
    p100 = ((p0)*(1.0-p1)*(1.0-p2)).astype('float32')
    p110 = ((p0)*(p1)*(1.0-p2)).astype('float32')
    p101 = ((p0)*(1.0-p1)*(p2)).astype('float32')
    p111 = ((p0)*(p1)*(p2)).astype('float32')
    
                
    # sample 8 times (all combinations of floor and ceiling)
    # input shape
    nI = list(I.shape)
    nIravel = [nI[0]*nI[1]*nI[2]]
    nIravel.extend(nI[3:]) # extend to accomodate all dimensions (assume that functions are at least of 1 dim)
    
    # output shape
    n = list(X0s.shape)
    n.extend(nI[3:])
    #nravel = [n[0]*n[1]]
    #nravel.extend(n[2:])
    
    I_ = np.reshape(I,nIravel)
    numUnique = np.unique(I_)
    print("numUnique is ")
    print(numUnique)
    print("shape of p and x000")
    print(p000.shape)
    print(X000.shape)
    retLabels = np.zeros((X000.shape[0],2)) # store arg max in second and linear in first
    #print(retLabels.shape)
    #retVals = np.zeros(X000.shape) # store max linear interpolation for labels
    I_labOnly = []
    I000 = []
    I010 = []
    I001 = []
    I100 = []
    I101 = []
    I011 = []
    I111 = []
    Itot = []
    for label in numUnique:
        print("label is " + str(label))
        I_labOnly = I_ == label
        I_labOnly = I_labOnly.astype('int') # potentially store as int?
        I000 = I_labOnly[X000]
        I010 = I_labOnly[X010]
        I001 = I_labOnly[X001]
        I100 = I_labOnly[X100]
        I110 = I_labOnly[X110]
        I101 = I_labOnly[X101]
        I011 = I_labOnly[X011]
        I111 = I_labOnly[X111]

        Itot = (I000*p000 + I010*p010 + I001*p001 + I100*p100 + I110*p110 + I101*p101 + I011*p011 + I111*p111).astype('float32')
        retLabels[Itot > retLabels[:,0],1] = label
        retLabels[:,0] = np.maximum(retLabels[:,0], Itot)
    
    retLabels = np.reshape(retLabels[:,1],n)
    
    # assume bc='nearest'

    return retLabels
###########################################################################
# Function Wrappers 

# Apply a function to set of points
# Note: if function does not have value assigned to desired coordinate, assign "N/A"
def applyFunction(coords, fun, d0, d1, d2, NN=0):
    '''
    Args:
    coords = spatial coordinates (in form of domain of fun) for which to return values for (assume 3D) (XxYxZ x 3 matrix)
    fun = function from domain of coordinates (3D) to range (any number of dimensions)
    domain = coordinates at which the function values are defined (in physical space) (XxYxZx3)

    Returns:
    coordVals = value of coords according to fun
    '''
    # coordVals = np.zeros((coords.shape[0], coords.shape[1], coords.shape[2], numFunValues))
    if (NN == 1):
        coordVals = interp3NN(d0,d1,d2,fun,coords[...,0], coords[...,1], coords[...,2])
    elif (NN == 2):
        coordVals = interp3Mode(d0,d1,d2,fun,coords[...,0], coords[...,1], coords[...,2])
    else:
        coordVals = interp3(d0, d1, d2, fun, coords[...,0], coords[...,1], coords[...,2])
    # Alternative interpret function that interprets based on NN rather than averaging 

    
    return coordVals

##############################################################3
# Loading Functions 

# Load Segmentation and recast as floating point
def loadSeg(fileHdr, fileImg):
    fileMap = nib.AnalyzeImage.make_file_map()
    fileMap['image'].fileobj = fileImg
    fileMap['header'].fileobj = fileHdr
    I = nib.AnalyzeImage.from_file_map(fileMap)
    
    # Get rid of extra dimensiosn if exist
    I = nib.funcs.squeeze_image(I)
    I.set_data_dtype(np.float64)
    
    return I

# Written to put Eileen's segmentation labels in correct form for interpolation (assume merged but not upsampled image)
# Ex. '/cis/home/kstouff4/Flattening_Notebook/MergedSegmentations/2_totHippo_dugOut_noUp.hdr', '/cis/home/kstouff4/Flattening_Notebook/MergedSegmentations/2_totHippo_dugOut_noUp.img'
def getKrimerLabels(mriHeader, mriImage):
    '''
    Args:
    mriHeader = filepath of mriHeader
    mriImage = filepath of mriImage
    
    returns Lx = function of R^3 to N
    x0L, x1L, x2L = indications of each of the x at which Lx is defined (in terms of pixel locations)
    '''
    I = loadSeg(mriHeader,mriImage)
    Lx = np.asanyarray(I.dataobj)
 
    d0 = 0.2273
    d1 = 0.2273
    d2 = 0.2304
    
    # origin at corner
    x0L = np.arange(Lx.shape[0],dtype='float32')*d0 # scale to appropriate pixel to tissue size
    x2L = np.arange(Lx.shape[2],dtype='float32')*d2 
    x1L = np.arange(Lx.shape[1],dtype='float32')*d1 

    X0L, X1L, X2L = np.meshgrid(x0L,x1L,x2L,indexing='ij')
    XL = np.stack((X0L,X1L,X2L),axis=-1)
    
    return Lx, XL, x0L, x1L, x2L

def getLabels(mriHeader, mriImage, d0, d1, d2):
    '''
    Args:
    mriHeader = filepath of mriHeader
    mriImage = filepath of mriImage
    
    returns Lx = function of R^3 to N
    x0L, x1L, x2L = indications of each of the x at which Lx is defined (in terms of pixel locations)
    '''
    I = loadSeg(mriHeader,mriImage)
    Lx = np.asanyarray(I.dataobj)
    print(I.affine)
    
    # origin at corner
    x0L = np.arange(Lx.shape[0],dtype='float32')*d0 # scale to appropriate pixel to tissue size
    x2L = np.arange(Lx.shape[2],dtype='float32')*d2 
    x1L = np.arange(Lx.shape[1],dtype='float32')*d1 

    X0L, X1L, X2L = np.meshgrid(x0L,x1L,x2L,indexing='ij')
    XL = np.stack((X0L,X1L,X2L),axis=-1)
    
    return Lx, XL, x0L, x1L, x2L
    
#######################################################################
# Deformation Functions 

# Deform krimer labels (from volume to volume)
def deformKrimerToExVivo(brainNum,coords,saveEnd,kanami=False):
    '''
    Assume matrix in Params per brain indicating Krimer surface to the brain in Mai
    
    Transpose matrix, then invert, and transform brain coordinates 
    '''
    
    krimerMatFile = '/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Params/Krimer_ERC+TEC_toBrain' + str(brainNum) + '.mat'
    if (kanami):
        krimerMatFile = '/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Params/Krimer_ERC+TEC_toBrain' + str(brainNum) + '_Kanami.mat'
    params = sp.io.loadmat(krimerMatFile, appendmat=False, struct_as_record=False)
    coordsRet = np.copy(coords)
    A = np.asarray(params['A']) # array of 4x4 matrices for each block
    
    # transpose matrix
    Aret = np.copy(A)
    Aret[0,0] = A[1,1]
    Aret[1,1] = A[0,0]
    Aret[0,1] = A[1,0]
    Aret[1,0] = A[0,1]
    Aret[0,2] = A[1,2]
    Aret[1,2] = A[0,2]
    Aret[0,3] = A[1,3]
    Aret[1,3] = A[0,3]
    Aret[2,0] = A[2,1]
    Aret[2,1] = A[2,0]
    
    Ainv = np.linalg.inv(Aret)
    coordsRet = np.zeros_like(coords)
    coordsRet[...,0] = Ainv[0,0]*coords[...,0] + Ainv[0,1]*coords[...,1] + Ainv[0,2]*coords[...,2] + Ainv[0,3]
    coordsRet[...,1] = Ainv[1,0]*coords[...,0] + Ainv[1,1]*coords[...,1] + Ainv[1,2]*coords[...,2] + Ainv[1,3]
    coordsRet[...,2] = Ainv[2,0]*coords[...,0] + Ainv[2,1]*coords[...,1] + Ainv[2,2]*coords[...,2] + Ainv[2,3]
   
    Lx, XL, x0L, x1L, x2L = getKrimerLabels('/cis/home/kstouff4/Documents/SueSurfaces/allMergeKrimer.hdr','/cis/home/kstouff4/Documents/SueSurfaces/allMergeKrimer.img')
    if (kanami):
        Lx, XL, x0L, x1L, x2L = getLabels('/cis/home/kstouff4/Documents/SueSurfaces/Krimer/Kanami/ERC+TEC_Kanami.hdr','/cis/home/kstouff4/Documents/SueSurfaces/Krimer/Kanami/ERC+TEC_Kanami.img',0.256,0.256,0.254) # official is 0.2557=x,0.2561=y,0.254=z
    krimer = applyFunction(coordsRet, Lx, x0L, x1L, x2L, NN=2)
    np.save('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Krimer/krimerLabels_' + saveEnd + '.npy',krimer)
   
    return krimer

def deformERCSurfaceToVTK(brainNum):
    if (brainNum == 2):
        byuCoords = sp.io.loadmat('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain2/Krimer/erc_brain2_redone_toMai_0803.mat',appendmat=False,struct_as_record=False)
    elif (brainNum == 5):
        byuCoords = sp.io.loadmat('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain5/Krimer/erc_brain5_toMai_0803.mat',appendmat=False,struct_as_record=False)

        
    coords = np.asarray(byuCoords['YXZ'])
    polys = np.asarray(byuCoords['polys'])
    krimer = deformKrimerToExVivo(brainNum,coords,'ercSurface_0803')
    
    # write vtk file with labels
    feats = []
    featsName = []
    feats.append(krimer)
    featsName.append('KrimerLabels')
    vt.writeVTK(coords,feats,featsName,'/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Krimer/ercSurface_0803_labels_brain' + str(brainNum) + '.vtk',polys)
    
    # save labels with coords as npz, YXZ coords and krimer labels 
    np.savez('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Krimer/ercSurface_0803_labels_brain' + str(brainNum) + '.npz',YXZ=coords,krimer=krimer,polys=polys)
    return

def modeColumns(numPointSurface,krimerCols,numCols=21,brain4=False):
    '''
    select label not equal to 0 if occurs 
    '''
    if (brain4):
        smallNum = 144 # old = 90
        largeNum = 315 # old = 359
        print(numPointSurface)
        print(smallNum+largeNum)
        totalLabels = np.zeros((numPointSurface,numCols)) # 21 is the length
        for i in range(numCols):
            totalLabels[0:smallNum,i] = np.squeeze(krimerCols[i*smallNum:(i+1)*smallNum])
        for i in range(numCols):
            totalLabels[smallNum:,i] = np.squeeze(krimerCols[21*smallNum+i*largeNum:21*smallNum+(i+1)*largeNum])
    else:
        totalLabels = np.zeros((numPointSurface,numCols)) # 21 is the length
        for i in range(numCols):
            totalLabels[:,i] = np.squeeze(krimerCols[i*numPointSurface:(i+1)*numPointSurface])
      
    #modeTot = sp.stats.mode(totalLabels,axis=-1)[0]
    modeTot = np.zeros((numPointSurface,1))
    for i in range(numPointSurface):
        counts = np.bincount(totalLabels[i,:].astype(int),minlength=10) # 10 Krimer labels
        modeHigh = np.argmax(counts)
        if (modeHigh == 0 and np.sum(counts[1:]) > 0):
            modeTot[i] = np.argmax(counts[1:])+1 # choose second highest mode above 0 only if there are other nonzero numbers
        else:
            modeTot[i] = modeHigh
   
    return np.squeeze(modeTot)

def deformERCLayerToVTK(vtkFileCols, vtkFileBot,brainNum,kanami):
    '''
    Args:
    vtkFileCols = vtkFile with cortical columns
    vtkFileBot = template file with bottom surface (i.e. template_0803.vtk)
    '''
    # get coordinates
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkFileCols)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    array = points.GetData()
    numpy_points = vtk_to_numpy(array)
    YXZ = np.copy(numpy_points)
    if ((brainNum != 2 and brainNum != 5) or kanami):
        YXZ[:,0] = numpy_points[:,1]
        YXZ[:,1] = numpy_points[:,0]
    krimerCols = deformKrimerToExVivo(brainNum,YXZ,'ercTop_Template_0803_Cols',kanami)
    
    # write vtk file with labels
    feats = []
    featsName = []
    feats.append(krimerCols)
    featsName.append('KrimerLabels')
    vt.writeVTK(YXZ,feats,featsName,'/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Krimer/ercTop_Template_0803_Colslabels_brain' + str(brainNum) + '.vtk')
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtkFileBot)
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    polydata = reader.GetOutput()
    
    points = polydata.GetPoints()
    array = points.GetData()
    numpy_points = vtk_to_numpy(array)

    
    krimer = modeColumns(numpy_points.shape[0],krimerCols,21,brainNum==4)
    dictToSave = dict()
    dictToSave['krimer'] = krimer
    sp.io.savemat('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Krimer/ercTop_Template_0803_modeLabels_brain' + str(brainNum) + '.mat',dictToSave)
    krimerArray = numpy_to_vtk(krimer)
    krimerArray.SetName('KrimerLabels')
    polydata.GetPointData().AddArray(krimerArray)

    writer = vtk.vtkPolyDataWriter()
    #writer = vtk.vtkBYUWriter()
    writer.SetFileName('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Krimer/ercTop_Template_0803_modeLabels_brain' + str(brainNum) + '.vtk')
    writer.SetInputData(polydata)
    writer.Update()
    writer.Write()
    return

def deformLabelsToSurface(brainNum,rigidMat,coordsReg,coordsSurf,labelVol,d0,d1,d2,savebase,excludeBack=False,rigidMat2=None):
    '''
    Args:
    rigidMat = template (with labels) to target (given brain structure)
    coordsReg = 3D coordinates (list = YxXxZx3) to get weighted mode labels at (send in filename base)
    coordsSurf = 3D coordinates of surface vertices to get weighted mode at # assume in YXZ 
    labelVol = all merge image for labels
    '''
    
    # in MRI Space (regular coordinates)
    regLx,coords,_,_,_ = getLabels(coordsReg+'.hdr',coordsReg+'.img',0.125,0.125,0.125)
    
    # matrix to move Kanami to MRI; if matrix takes Kanami to MRI in Mai, apply rigidMat2 first (MRI to Mai)
    params = sp.io.loadmat(rigidMat, appendmat=False, struct_as_record=False)
    coordsRet = np.copy(coords)
    A = np.asarray(params['A']) # array of 4x4 matrices for each block
    
    # transpose matrix
    Aret = np.copy(A)
    Aret[0,0] = A[1,1]
    Aret[1,1] = A[0,0]
    Aret[0,1] = A[1,0]
    Aret[1,0] = A[0,1]
    Aret[0,2] = A[1,2]
    Aret[1,2] = A[0,2]
    Aret[0,3] = A[1,3]
    Aret[1,3] = A[0,3]
    Aret[2,0] = A[2,1]
    Aret[2,1] = A[2,0]
    
    Ainv = np.linalg.inv(Aret)
    if (rigidMat2 is not None):
        paramsFirst = sp.io.loadmat(rigidMat2,appendmat=False,struct_as_record=False)
        A = np.asarray(paramsFirst['A'])
        # transpose 
        Aret = np.copy(A)
        Aret[0,0] = A[1,1]
        Aret[1,1] = A[0,0]
        Aret[0,1] = A[1,0]
        Aret[1,0] = A[0,1]
        Aret[0,2] = A[1,2]
        Aret[1,2] = A[0,2]
        Aret[0,3] = A[1,3]
        Aret[1,3] = A[0,3]
        Aret[2,0] = A[2,1]
        Aret[2,1] = A[2,0]
        coordsRet = np.zeros_like(coords)
        coordsRet[...,0] = Aret[0,0]*coords[...,0] + Aret[0,1]*coords[...,1] + Aret[0,2]*coords[...,2] + Aret[0,3]
        coordsRet[...,1] = Aret[1,0]*coords[...,0] + Aret[1,1]*coords[...,1] + Aret[1,2]*coords[...,2] + Aret[1,3]
        coordsRet[...,2] = Aret[2,0]*coords[...,0] + Aret[2,1]*coords[...,1] + Aret[2,2]*coords[...,2] + Aret[2,3]
        coordsRet2 = np.zeros_like(coords)
        coordsRet2[...,0] = Ainv[0,0]*coordsRet[...,0] + Ainv[0,1]*coordsRet[...,1] + Ainv[0,2]*coordsRet[...,2] + Ainv[0,3]
        coordsRet2[...,1] = Ainv[1,0]*coordsRet[...,0] + Ainv[1,1]*coordsRet[...,1] + Ainv[1,2]*coordsRet[...,2] + Ainv[1,3]
        coordsRet2[...,2] = Ainv[2,0]*coordsRet[...,0] + Ainv[2,1]*coordsRet[...,1] + Ainv[2,2]*coordsRet[...,2] + Ainv[2,3]
        coordsRetMai = np.copy(coordsRet)
        coordsRet = np.copy(coordsRet2)
    else:
        coordsRet = np.zeros_like(coords)
        coordsRet[...,0] = Ainv[0,0]*coords[...,0] + Ainv[0,1]*coords[...,1] + Ainv[0,2]*coords[...,2] + Ainv[0,3]
        coordsRet[...,1] = Ainv[1,0]*coords[...,0] + Ainv[1,1]*coords[...,1] + Ainv[1,2]*coords[...,2] + Ainv[1,3]
        coordsRet[...,2] = Ainv[2,0]*coords[...,0] + Ainv[2,1]*coords[...,1] + Ainv[2,2]*coords[...,2] + Ainv[2,3]
    
    Lx, XL, x0L, x1L, x2L = getLabels(labelVol+'.hdr',labelVol + '.img',d0,d1,d2)
    labsInVol = applyFunction(coordsRet, Lx, x0L, x1L, x2L, NN=2)
    maskedLabsInVol = labsInVol*(regLx > 0) # mask out those that are not in the volume (become background)
    print("masked Labs in Vol")
    print(np.unique(maskedLabsInVol))
    
    np.savez('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Labels/' + savebase + '_vol.npz',allVolLabs=labsInVol,maskedVolLabs=maskedLabsInVol,coords=coords)
    numLabs = np.unique(labsInVol).shape[0]
    
    # select all points in region
    labsInVolrav = np.ravel(labsInVol)
    regLxrav = np.ravel(regLx)
    subsetPointsLab = labsInVolrav[regLxrav > 0].astype(int)
    x0rav = np.ravel(coords[...,0])[regLxrav > 0]
    x1rav = np.ravel(coords[...,1])[regLxrav > 0]
    x2rav = np.ravel(coords[...,2])[regLxrav > 0]
    
    # if rigidMat, assume surface is in Mai
    if (rigidMat2 is not None):
        x0rav = np.ravel(coordsRetMai[...,0])[regLxrav > 0]
        x1rav = np.ravel(coordsRetMai[...,1])[regLxrav > 0]
        x2rav = np.ravel(coordsRetMai[...,2])[regLxrav > 0]

    
    # tallies
    tallies = np.zeros((coordsSurf.shape[0],numLabs)) # assume 0 is background
    for p in range(x0rav.shape[0]):
        dS = (x0rav[p]-coordsSurf[...,0])**2 + (x1rav[p]-coordsSurf[...,1])**2 + (x2rav[p]-coordsSurf[...,2])**2
        tallies[np.argmin(dS),subsetPointsLab[p]]+=1
    talliesF = np.argmax(tallies,axis=-1)
    # work around
    if (excludeBack):
        for p in range(tallies.shape[0]):
            if (talliesF[p] == 0 and np.sum(tallies[p,1:]) > 0):
                talliesF[p] = np.argmax(tallies[p,1:])+1
    dictToSave = dict()
    dictToSave['labels'] = talliesF
    if (excludeBack):
        sp.io.savemat('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Labels/' + savebase+'_onSurf_projectOutExcludeBack.mat',dictToSave)
    else:
        sp.io.savemat('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Labels/' + savebase+'_onSurf_projectOut.mat',dictToSave)

    

    # assume coords are equally sampled and get weighted mode of labs in Vol onto surface (alternative is for each surface point to find NN)
    labsOnSurf = applyFunction(coordsSurf,labsInVol,np.asarray([coords[0,0,0,0],coords[1,0,0,0]]),np.asarray([coords[0,0,0,1],coords[0,1,0,1]]),np.asarray([coords[0,0,0,2],coords[0,0,1,2]]),NN=2)
    dictToSave = dict()
    dictToSave['labels'] = labsOnSurf
    sp.io.savemat('/cis/home/kstouff4/Documents/HistoMRIPipeline/Brain' + str(brainNum) + '/Labels/' + savebase+'_onSurf_weightedMode.mat',dictToSave)

    return labsOnSurf,talliesF

# Deform MRI Image to Particular Mai space (slice) given by coords in npy file 
def deformMRItoMai(hdr,img,brainNum,maiNPY,savename,seg=False,x=None,y=None,z=None,down=1):
    coordsMai = np.load(maiNPY)
    I,XI,x0I,x1I,x2I = getLabels(hdr,img,0.125*down,0.125*down,0.125*down) # in corner 
    if (brainNum != 2):
        x0I = x0I - np.mean(x0I)
        x1I = x1I - np.mean(x1I)
        x2I = x2I - np.mean(x2I)
        X0,X1,X2 = np.meshgrid(x0I,x1I,x2I,indexing='ij')
        XI = np.stack((X0,X1,X2),axis=-1)
    if (seg):
        maiVals = applyFunction(coordsMai, I, x0I, x1I, x2I, NN=2) # mode if for segmentations
    else:
        maiVals = applyFunction(coordsMai, I, x0I, x1I, x2I, NN=0) # linear for MRI
    np.save(savename+'.npy',maiVals)
    
    if (x is not None and y is not None):
        f,ax = plt.subplots()
        if (not seg and down == 1):
            ax.imshow(maiVals,cmap='gray',extent=(x[0],x[1],y[0],y[1]),origin='lower')
        elif (down > 1):
            alphaMat = (maiVals < 0).astype(float) 
            print(alphaMat.shape)
            print(np.unique(alphaMat))
            print(maiVals.shape)
            cmapK = cm.get_cmap('jet')
            cmapK.set_under('w',alpha=0)
            im = ax.imshow(maiVals,cmap=cmapK,extent=(x[0],x[1],y[0],y[1]),origin='lower',vmin=0)
            c = f.colorbar(im,ax=ax)
            c.set_label('Tau Tangles / mm^2')
        else:
            
            colorsN = ['white','yellow', 'green','pink', 'red', 'orange', 'plum', 'blue', 'lime', 'cyan', 'indigo', 'darkorchid', 'fuchsia']    
            cmapNew = colors.ListedColormap(colorsN)
            bounds=[-0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5]
            norm = colors.BoundaryNorm(bounds, cmapNew.N)
            im = ax.imshow(maiVals,cmap=cmapNew,extent=(x[0],x[1],y[0],y[1]),origin='lower')
            c = f.colorbar(im,ax=ax)
            c.set_ticks([0,1,2,3,4,5,6,7,8,9,10,11,12])
            c.set_ticklabels(['Back','Alv','Amyg','CA1','CA2','CA3','ERC','DG_G','Hil','DG_M','ParaS','PreS','Sub'])
        ax.set_aspect('equal')
        ax.set_xlabel('X-axis (Med--Lat) (mm)')
        ax.set_ylabel('Y-axis (Inf--Sup) (mm)')
        ax.set_title('Mai Slice at ' + str(z) + ' mm')
        f.savefig(savename+'.png',dpi=300)
    return
