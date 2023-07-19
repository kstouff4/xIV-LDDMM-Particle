import numpy as np
import torch

import nibabel as nib
import pandas as pd
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from matplotlib import pyplot as plt

def applyAffine(Z, nu_Z, A, tau):
    '''
    Makes a new set of particles based on an input set and applying the affine transformation given by matrix A and translation, tau
    '''
    R = torch.clone(Z)
    nu_R = torch.clone(nu_Z)
    R = R@A.T + tau
    return R,nu_R

def readFromPrevious(npzFile):
    '''
    Assumes source deformed to target previously.
    Initializes new source as deformed source.
    '''
    npz = np.load(npzFile)
    S = torch.tensor(npz['D'])
    nu_S = torch.tensor(npz['nu_D'])
    T = torch.tensor(npz['T'])
    nu_T = torch.tensor(npz['nu_T'])
    
    return S,nu_S,T,nu_T

def getFromFile(npzFile):
    npz = np.load(npzFile)
    print("min and max")
    print(np.min(npz[npz.files[0]],axis=0))
    print(np.max(npz[npz.files[0]],axis=0))
    S = torch.tensor(npz[npz.files[0]])
    nu_S = torch.tensor(npz[npz.files[1]])
    return S,nu_S

def makeFromSingleChannelImage(imageFile,resXYZ,bg=[0],ordering=None,ds=1,axEx=None,rotate=False,flip=False,weights=None):
    '''
    Makes discrete particle representation from image file (NIFTI or ANALYZE).
    Assumes background has value 0 and excluded as no data.
    
    Centers particles around the origin based on bounding box of coordinates.
    ds = amount to downsample by
    axEx = tuple (axis along which to select one plane and plane number to select)
    '''
    
    imInfo = nib.load(imageFile)
    im = np.squeeze(np.asanyarray(imInfo.dataobj)).astype('float32')
    if (axEx is not None):
        if (axEx[0] == 0):
            im = np.squeeze(im[axEx[1],...])
        elif (axEx[0] == 1):
            im = np.squeeze(im[:,axEx[1],...])
        elif (axEx[0] == 2):
            im = np.squeeze(im[:,:,axEx[1],...])
    # rotate such that second axis is longer than first
    if (rotate):
        if (im.shape[1] < im.shape[0]):
            im = np.swapaxes(im,0,1)
    if (flip):
        im = np.flip(im,axis=0)
    dims = im.shape
    if (ds > 1):
        if len(dims) == 2:
            im = im[0::ds,0::ds]
        elif len(dims) == 3:
            im = im[0::ds,0::ds,0::ds]
    dims = im.shape
    print("dims is ", dims)
    x0 = np.arange(dims[0])*resXYZ
    x0 -= np.mean(x0)
    x1 = np.arange(dims[1])*resXYZ
    x1 -= np.mean(x1)
    if len(dims) > 2:
        x2 = np.arange(dims[2])*resXYZ
        x2 -= np.mean(x2)
    else:
        x2 = np.zeros(1) # default to centering 2d image at 0
    
    X,Y,Z = torch.meshgrid(torch.tensor(x0),torch.tensor(x1),torch.tensor(x2),indexing='ij')
    S = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=-1)
    print("size of S:", S.shape)
    listOfNu = []
    
    if (ordering is not None):
        uniqueVals = ordering
    else:
        uniqueVals = np.unique(im)
        if (bg is not None):
            for bbg in bg:
                uniqueVals = uniqueVals[uniqueVals != bbg]
        
    numUnique = len(uniqueVals)

    keepSum = torch.zeros((S.shape[0],1))
    for u in range(len(uniqueVals)):
        n = torch.tensor((im == uniqueVals[u]))
        listOfNu.append(n.flatten())
        keepSum += n.flatten()[...,None]
    toKeep = torch.squeeze(keepSum > 0)
    listOfNewNu = []
    for l in listOfNu:
        listOfNewNu.append(l[toKeep])
    nu_S = torch.stack(listOfNewNu,axis=-1)

    #toKeep = nu_S.sum(axis=-1) > 0
    S = S[toKeep]
    #nu_S = nu_S[toKeep]
    
    if (weights is not None):
        nu_S = nu_S*weights

    return S,nu_S

def combineObjects(Slist,nu_Slist,d=3):
    '''
    combine list of particles that have different feature values; assume mutually exclusive lists
    '''
    nu_Stot = 0
    sTot = 0
    for i in range(len(Slist)):
        nu_Stot += nu_Slist[i].shape[-1]
        sTot += Slist[i].shape[0]
    Scomb = torch.zeros((sTot,d))
    nuScomb = torch.zeros((sTot,nu_Stot))
    cntS = 0
    cntnuS = 0
    for i in range(len(Slist)):
        Scomb[cntS:cntS+Slist[i].shape[0],...] = Slist[i]
        nuScomb[cntS:cntS+Slist[i].shape[0],cntnuS:cntnuS+nu_Slist[i].shape[-1]] = nu_Slist[i]
        cntS += Slist[i].shape[0]
        cntnuS += nu_Slist[i].shape[-1]
    return Scomb,nuScomb
        

def readParticleApproximation(particleNPZ):
    '''
    If in the form of particle approximation, then spatial coordinates will be saved as "Z" and features as "nu_Z"
    '''
    npz = np.load(particleNPZ)
    S = torch.tensor(npz['Z'])
    nu_S = torch.tensor(npz['nu_Z'])

    return S,nu_S

def readSpaceFeatureCSV(coordCSV,coordNames,featCSV,featNames,scale=None,labs=None):
    '''
    For reading in a csv with each row representative of measure (e.g. cell or single mRNA)
    Scale datapoints to mm if in different coordinates (e.g. microns --> scale = 1e-3)
    Center data points around 0,0
    '''
    df_s = pd.read_csv(coordCSV)
    df_f = pd.read_csv(featCSV)
    if (len(featNames) > 1):
        nu_S = torch.tensor(df_f[featNames].values)
    elif labs is not None:
        listOfNu = []
        nu_S_single = df_f[featNames].values
        if (np.min(nu_S_single) == 1):
            nu_S_single -= 1
        for u in range(labs):
            n = torch.tensor((nu_S_single == u))
            listOfNu.append(n.flatten())
        nu_S = torch.stack(listOfNu,axis=-1)
    else:
        nu_S_single = df_f[featNames].values
        uVals = np.unique(nu_S_single)
        listOfNu = []
        for u in uVals:
            n = torch.tensor((nu_S_single == u))
            listOfNu.append(n.flatten())
        nu_S = torch.stack(listOfNu,axis=-1)
        
    S = torch.tensor(df_s[coordNames].values)
    
    if scale is not None:
        S = torch.tensor(scale*df_s[coordNames].values)
    S = S - torch.mean(S,axis=0)
    if (S.shape[-1] < 3):
        S = torch.cat((S,S[...,0][...,None]*0),-1)
    return S,nu_S

def returnMultiplesSpace(S,nu_S,k):
    '''
    Add particles for testing by replicating each particle Si, nu_Si k times
    '''
    N = S.shape[0]
    Sk = torch.zeros((N*k,S.shape[-1]))
    nu_Sk = torch.zeros((N*k,nu_S.shape[-1]))
    
    for i in range(k):
        Sk[i*N:(i+1)*(N),:] = S
        nu_Sk[i*N:(i+1)*N,:] = nu_S/torch.tensor(k)
    return Sk,nu_Sk

def makeBinsFromMultiChannelImage(imageFile,res,dimEff,dimFeats,ds=1,z=0,threshold=0,bins=1,reverse=False):
    imageSuff = imageFile.split('.')[-1] 
    if (imageSuff == 'tif' or imageSuff == 'tif' or imageSuff == 'png' or imageSuff == 'TIF'):
        im = np.squeeze(plt.imread(imageFile))
    else:
        im = nib.load(imageFile)
        im = np.squeeze(im.dataobj)
    
    imDS = im[0::ds,...]
    imDS = imDS[:,0::ds,...]
    if dimEff == 3:
        imDS = imDS[:,:,0::ds,...]
        if dimFeats == 1 and imDS.shape[-1] > 1:
            imDS = imDS[...,None]
    
    # make grid
    dims = imDS.shape
    print("dims is ", dims)
    x0 = np.arange(dims[0])*res[0]
    x0 -= np.mean(x0)
    x1 = np.arange(dims[1])*res[1]
    x1 -= np.mean(x1)
    if dimEff > 2:
        x2 = np.arange(dims[2])*res[2]
        x2 -= np.mean(x2)
    else:
        x2 = np.zeros(1) + z
    
    X,Y,Z = torch.meshgrid(torch.tensor(x0),torch.tensor(x1),torch.tensor(x2),indexing='ij')
    S = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=-1)
    print("S shape, ", S.shape)
    
    # use tissue volume as weight
    if bins > 0:
        nu_S = torch.zeros((S.shape[0],(bins-threshold)*dimFeats))
        for di in range(dimFeats):
            nu_Sdi = np.ravel(imDS[...,di]).astype('float32')
            if (reverse):
                nu_Sdi = -1*nu_Sdi
            b = np.arange(bins)*(np.max(nu_Sdi)+1-np.min(nu_Sdi))/bins + np.min(nu_Sdi) - 0.5
            print("bins are: ", b)
            nu_Sdibin = np.ravel(np.digitize(nu_Sdi,b)-1) # 1 based
            print(np.unique(nu_Sdibin))
            oneHot = np.zeros((nu_Sdibin.shape[0],bins))
            print("one hot shape, ", oneHot.shape)
            oneHot[np.arange(nu_Sdibin.shape[0]),nu_Sdibin] = 1.0
            if (threshold > 0):
                oneHot = oneHot[:,threshold:]
            nu_S[:,di*(bins-threshold):(di+1)*(bins-threshold)] = torch.tensor(oneHot)*torch.tensor(np.prod(res)) # set weights to tissue area of each feature (if multichannel, then total weights are multiplied by that
    else:
        nu_S = torch.zeros((S.shape[0],dimFeats))
        for di in range(dimFeats):
            nu_S[:,di] = torch.tensor(imDS[...,di]).flatten()
    
    if threshold > 0:
        keep = torch.sum(nu_S,axis=-1) > 0
        nu_S = nu_S[keep,...]
        S = S[keep,...]
    
    return S,nu_S  

def makeBinsFromDistance(imageFile,res,dimEff,targetFeat,ds=1,z=0,threshold=0,bins=10,weights=None):
    '''
    Categorize values of intensity based on distance from particular color; reduces multiple feature dimensions to 1
    
    targetFeat should be of form (...) of size F
    '''
    imageSuff = imageFile.split('.')[-1] 
    if (imageSuff == 'tif' or imageSuff == 'tif' or imageSuff == 'png' or imageSuff == 'TIF'):
        im = np.squeeze(plt.imread(imageFile))
    else:
        im = nib.load(imageFile)
        im = np.squeeze(im.dataobj)
    
    imDS = im[0::ds,...]
    imDS = imDS[:,0::ds,...]
    if dimEff == 3:
        imDS = imDS[:,:,0::ds,...]
        if dimFeats == 1 and imDS.shape[-1] > 1:
            imDS = imDS[...,None]
    
    # make grid
    dims = imDS.shape
    print("dims is ", dims)
    x0 = np.arange(dims[0])*res[0]
    x0 -= np.mean(x0)
    x1 = np.arange(dims[1])*res[1]
    x1 -= np.mean(x1)
    if dimEff > 2:
        x2 = np.arange(dims[2])*res[2]
        x2 -= np.mean(x2)
    else:
        x2 = np.zeros(1) + z
    
    X,Y,Z = torch.meshgrid(torch.tensor(x0),torch.tensor(x1),torch.tensor(x2),indexing='ij')
    S = torch.stack((X.flatten(),Y.flatten(),Z.flatten()),axis=-1)
    print("S shape, ", S.shape)

    repValue = np.zeros_like(imDS)
    repValue[...,0:] = np.asarray(targetFeat)
    print("repValue shape, ", repValue.shape)
    
    distS = (imDS-repValue)**2
    
    # weigh each dimension differently
    if weights is not None:
        for i in range(distS.shape[-1]):
            distS[...,i] = distS[...,i]*weights[i]
    
    dist = np.sqrt(np.sum(distS,axis=-1))
    print("dist shape, ", dist.shape)
    nu_S = torch.zeros((S.shape[0],(bins-threshold)))
    nu_Sdi = -1.0*np.ravel(dist).astype('float32') # largest value (largest bin) = closest to value 
    b = np.arange(bins)*(np.max(nu_Sdi)+1-np.min(nu_Sdi))/bins + np.min(nu_Sdi) - 0.5
    print("bins are: ", b)
    nu_Sdibin = np.ravel(np.digitize(nu_Sdi,b)-1) # 1 based
    print(np.unique(nu_Sdibin))
    oneHot = np.zeros((nu_Sdibin.shape[0],bins))
    print("one hot shape, ", oneHot.shape)
    oneHot[np.arange(nu_Sdibin.shape[0]),nu_Sdibin] = 1.0
    if (threshold > 0):
        oneHot = oneHot[:,threshold:]
    nu_S[:,0:] = torch.tensor(oneHot)*torch.tensor(np.prod(res)) # set weight to tissue area
    
    if threshold > 0:
        keep = torch.sum(nu_S,axis=-1) > 0
        nu_S = nu_S[keep,...]
        S = S[keep,...]
    
    return S,nu_S  


    