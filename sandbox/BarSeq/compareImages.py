import sys
from sys import path as sys_path
sys_path.append('../')
sys_path.append('../../')
sys_path.append('../../xmodmap/')
sys_path.append('../../xmodmap/io/')
import shootParticles as sp
import saveState as ss
from analyzeOutput import interpolateWithImage
import initialize as init
import getInput as gI

import torch
np_dtype = "float32" #"float64"
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor #DoubleTensor 
else:
    dtype = torch.FloatTensor

import numpy as np
import nibabel as nib

def makeImages(S,nu_S,Td,nu_Td,sigma,featNamesS,featNamesT,savedir):
    # read in atlas points (high resolution)
    # read in deformed target points 
    # interpolate values onto image points and save as image 
    mins = np.min(S,axis=0) - sigma
    maxes = np.max(S,axis=0) + sigma
    x0 = np.arange(mins[0],maxes[0],sigma[0])
    x1 = np.arange(mins[1],maxes[1],sigma[1])
    x2 = np.arange(mins[2],maxes[2],sigma[2])
    X0,X1,X2 = np.meshgrid(x0,x1,x2,indexing='ij')
    print("image size is: ", X0.shape)
    Xg = np.stack((X0.ravel(),X1.ravel(),X2.ravel()),axis=-1)
    
    nu_Gs = sp.resampleGauss(S,nu_S,Xg,sigma[0])
    nu_Gsim = np.reshape(nu_Gs,(X0.shape[0],X0.shape[1],X0.shape[2],nu_Gs.shape[-1]))
        
    empty_header = nib.Nifti1Header()
    nu_GsMV = nib.Nifti1Image((np.argmax(nu_Gsim,axis=-1) + 1)*(np.sum(nu_Gsim,axis=-1)>0.001),np.eye(4),empty_header)
    nu_GsSum = nib.Nifti1Image(np.sum(nu_Gsim,axis=-1), np.eye(4), empty_header)
    
    nib.save(nu_GsMV,savedir + 'nu_Smax.nii.gz')
    nib.save(nu_GsSum,savedir + 'nu_Ssum.nii.gz')
    
    nu_Gt = sp.resampleGauss(Td,nu_Td,Xg,sigma[0])
    nu_Gtim = np.reshape(nu_Gt,(X0.shape[0],X0.shape[1],X0.shape[2],nu_Gt.shape[-1]))

    nu_GtSum = nib.Nifti1Image(np.sum(nu_Gtim,axis=-1), np.eye(4), empty_header)
    nu_GtMV = nib.Nifti1Image((np.argmax(nu_Gtim,axis=-1) + 1)*(np.sum(nu_Gtim,axis=-1)>0.001),np.eye(4),empty_header)
    
    nib.save(nu_GtSum,savedir + 'nu_Tsum.nii.gz')
    nib.save(nu_GtMV,savedir + 'nu_Tmax.nii.gz')
    
    tot = np.sum(nu_Gtim,axis=-1)
    denom = np.zeros_like(tot)
    denom[tot > 0] = 1.0/tot[tot > 0]
    if (featNamesT is not None):
        for f in range(nu_Gt.shape[-1]):
            nu_Gtf = nib.Nifti1Image(np.squeeze(nu_Gtim[...,f]*denom), np.eye(4), empty_header)
            nib.save(nu_Gtf,savedir + 'nu_Tf' + str(f) + '_' + featNamesT[f] + '.nii.gz')
    
    tot = np.sum(nu_Gsim,axis=-1)
    denom = np.zeros_like(tot)
    denom[tot > 0] = 1.0/tot[tot > 0]
    
    if (featNamesS is not None):
        for f in range(nu_Gs.shape[-1]):
            nu_Gtf = nib.Nifti1Image(np.squeeze(nu_Gsim[...,f]*denom), np.eye(4), empty_header)
            nib.save(nu_Gtf,savedir + 'nu_Sf' + str(f) + '_' + featNamesS[f] + '.nii.gz')


    return Xg,X0

def interpolateOnAtlasImage(TdRavel,TdGrid,savename):
    resAtlas = [0.01,0.01,0.01]
    imgS = '/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen_10_anno_16bit_ap_672.img'
    interpolateWithImage(imgS,resAtlas,TdRavel,TdGrid,savename,flip=False,ds=2)
    return
    

def main():
    sigma = 0.05 # 50 micron 
    atlasImage='/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/downFromOld__optimalZnu_ZAllwC1.0_sig0.2_Nmax1500.0_Npart2000.0.npz'
    origTarg = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/SliceToSlice/BarSeq/0.5/allSlices.npz'
    targetImage = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/BarSeq/AllenAtlas200ToBarSeq/output_dl_sig_its_albega_N-35_[0.2, 0.1, 0.05][0.5, 0.2, 0.05, 0.02]_100_0.1None_67827flipFullAtlas/testOutput_Dvars.npz'
    savedir = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/BarSeq/AllenAtlas200ToBarSeq/output_dl_sig_its_albega_N-35_[0.2, 0.1, 0.05][0.5, 0.2, 0.05, 0.02]_100_0.1None_67827flipFullAtlas/sig' + str(sigma) + 'gridIm_'
    
    featNamesT = ['Rab3c', 'Slc17a7', 'Nrsn1', 'Dgkb', 'Gria1']
    S,nu_S = gI.getFromFile(atlasImage)
    T,nu_T = gI.getFromFile(origTarg)
    T = T.cpu().numpy()
    nu_T = nu_T.cpu().numpy()
    
    jacFile = np.load(targetImage.replace('Dvars.npz','D10_jacobian.npz'))
    j = jacFile['j']
    
    # flip Allen atlas over z axis
    S[:,-1] = -1.0*S[:,-1]

    info = np.load(targetImage)
    Td = info['Td']
    nu_Td = info['nu_Td']
    S = S.cpu().numpy()
    nu_S = nu_S.cpu().numpy()
    
    D = info['D']
    nu_D = info['nu_D']
    nu_Dpi = info['nu_Dpi']
    
    Xgs,X0s = makeImages(S,nu_S,Td,nu_Td,[sigma,sigma,sigma],None,featNamesT,savedir+'CCF_')
    interpolateOnAtlasImage(Xgs,np.stack((X0s,X0s,X0s),axis=-1),savedir+'CCFimage_')
    Xg, X0 = makeImages(T,nu_T,D,nu_D,[sigma,sigma,sigma],featNamesT,None,savedir+'BarSeq_')
    
    nu_Gs = sp.resampleGauss(D,np.squeeze(j)[...,None],Xg,sigma)
    nu_Gsim = np.reshape(nu_Gs,(X0.shape[0],X0.shape[1],X0.shape[2],1))
        
    empty_header = nib.Nifti1Header()
    nu_GsMV = nib.Nifti1Image((np.squeeze(nu_Gsim)),np.eye(4),empty_header)
    
    nib.save(nu_GsMV,savedir + 'BarSeq_jacobianD.nii.gz')

    return

if __name__ == "__main__":
    main()
