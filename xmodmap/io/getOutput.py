import numpy as np
from matplotlib import pyplot as plt
import torch

def computeRegionStatisticsImage(npzFile,labels,plotOriginal=False):
    '''
    npz file should have nu_Scomb = original source compartments (should be 1 label per particle)
    nu_D = deformed source 
    labels = list of named labels per nu_Scomb
    '''
    info = np.load(npzFile)
    nu_D = info['nu_D']
    nu_Scomb = info['nu_Scomb']
    w_Scomb = np.sum(nu_Scomb,axis=-1)
    startW = np.sum(nu_Scomb,axis=0)
    wD = np.sum(nu_D,axis=-1)
    zeta_Scomb = nu_Scomb/w_Scomb[...,None]
    nu_Dcomb = zeta_Scomb*wD[...,None]
    D = info['D']
    nu_DcombRat = zeta_Scomb*(wD/w_Scomb)[...,None]
    
    imageNames = []
    imageVals = []
    imageNames.append('maxVal')
    imageVals.append(np.argmax(nu_Dcomb,axis=-1))
    imageNames.append('mass')
    imageVals.append(wD)
    imageNames.append('RatioInMass')
    imageVals.append(wD/w_Scomb)
    imageNames.append('PercentChangeInMass')
    imageVals.append(100.0*(wD - w_Scomb)/w_Scomb)
    for l in range(len(labels)):
        imageNames.append(labels[l])
        imageVals.append(nu_Dcomb[:,l])
    
    writeVTK(D,imageVals,imageNames,npzFile.replace('.npz','_regions.vtk'),polyData=None)
    writeVTK(info['S'],imageVals,imageNames,npzFile.replace('.npz','_regionsInOrigCoords.vtk'),polyData=None)
    f,ax = plt.subplots(3,1,figsize=(6,10))
    yErrMass = np.std(nu_Dcomb,axis=0)
    yMeanMass = np.sum(nu_Dcomb,axis=0)
    yErrRatio = np.std(nu_DcombRat,axis=0)
    yMeanRatio = np.sum(nu_DcombRat,axis=0)
    yPerChange = 100.0*(yMeanMass - startW)/startW
    
    labelsNew = []
    labelsNew.append('')
    for l in labels:
        labelsNew.append(l)
    ax[0].bar(np.arange(len(yErrMass)),yMeanMass)
    ax[0].set_ylabel('Total Mass (mm$^3$)')
    ax[0].set_xticklabels(labelsNew)
    ax[1].bar(np.arange(len(yErrRatio)),yMeanRatio)
    ax[1].set_ylabel('Ratio in Mass')
    ax[1].set_xticklabels(labelsNew)
    ax[2].bar(np.arange(len(yErrRatio)),yPerChange)
    ax[2].set_ylabel('Percent Change in Mass')
    ax[2].set_xticklabels(labelsNew)
    f.savefig(npzFile.replace('.npz','_regionsStats.png'),dpi=300)
    
    if plotOriginal:
        writeVTK(info['S'],[np.argmax(nu_Scomb,axis=-1),np.sum(nu_Scomb,axis=-1)],['MaxVal','TotalMass'],npzFile.replace('.npz','_origNu_Scomb.vtk'),polyData=None)
    return info['S'], nu_Dcomb

def analyzeLongitudinalMass(listOfNu,S,ages,savename,labels):
    '''
    Assume list is ordered in time
    Assume each particle is 100% one type
    '''
    X = np.ones((len(ages),2))
    X[:,1] = ages
    
    coef = np.linalg.inv(X.T@X)@X.T #
    Y = np.zeros((len(ages),listOfNu[0].shape[0])) # number of particles
    for l in range(len(ages)):
        Y[l,:] = np.sum(listOfNu[l],axis=-1)[None,...] # get total mass
    beta = coef@Y
    imageNames = []
    imageVals = []
    imageNames.append("SlopeEst_ChangeInMassPerYear")
    imageVals.append(beta[1,:].T)
    imageNames.append("SlopeEst_PercChangeInMassPerYear_OfStart")
    imageVals.append(100.0*beta[1,:].T/Y[0,:].T)
    imageNames.append("Perc_ChangeInMassPerYear")
    imageVals.append((100.0*(Y[-1,:]-Y[0,:])/((Y[0,:])*(ages[-1]-ages[0]))).T)
    imageNames.append("ChangeInMassPerYear")
    imageVals.append(((Y[-1,:]-Y[0,:])/((ages[-1]-ages[0]))).T)
    writeVTK(S,imageVals,imageNames,savename,polyData=None)
    
    f,ax = plt.subplots(len(labels),1,figsize=(6,12))
    slopes = beta[1,:]
    slopes = []
    for l in range(len(labels)):
        slopes.append(beta[1,np.squeeze(listOfNu[0][:,l] > 0)].T)
    for i in range(len(labels)):
        ax[i].hist(slopes[i],label="Mean $\pm$ Std = {0:.6f} $\pm$ {1:.6f}".format(np.mean(slopes[i]),np.std(slopes[i])))
        ax[i].set_xlabel('Estimated (LS) Change In Mass Per Year')
        ax[i].set_ylabel(labels[i] + ' Frequency')
        ax[i].set_xlim([-0.025,0.025])
        ax[i].legend()
    f.savefig(savename.replace('.vtk','_stats.png'),dpi=300)
    np.savez(savename.replace('.vtk','.npz'),beta=beta,coef=coef,Y=Y)
    return

def getJacobian(Di,nu_Si,nu_Di,savename):
    if torch.is_tensor(Di):
        D = Di.cpu().numpy()
        nu_S = nu_Si.cpu().numpy()
        nu_D = nu_Di.cpu().numpy()
    else:
        D = Di
        nu_S = nu_Si
        nu_D = nu_Di
    j = np.sum(nu_D,axis=-1)/np.sum(nu_S,axis=-1)
    imageNames = ['maxVal','totalMass','jacobian']
    imageVals = [np.argmax(nu_D,axis=-1)+1, np.sum(nu_D,axis=-1), j]
    writeVTK(D,imageVals,imageNames,savename,polyData=None)
    np.savez(savename.replace('.vtk','.npz'),j=j)
    return savename.replace('.vtk','.npz')

def splitZs(Ti,nu_Ti,Di,nu_Di,savename,units=10,jac=None):
    # split target, and deformed source along Z axis (last one) into units
    if torch.is_tensor(Ti):
        T = Ti.cpu().numpy()
        nu_T = nu_Ti.cpu().numpy()
    else:
        T = Ti
        nu_T = nu_Ti
    if torch.is_tensor(Di):
        D = Di.cpu().numpy()
        nu_D = nu_Di.cpu().numpy()
    else:
        D = Di
        nu_D = nu_Di
    ma = np.max(T,axis=0)[-1]
    mi = np.min(T,axis=0)[-1]
    mas = np.max(D,axis=0)[-1]
    mis = np.min(D,axis=0)[-1]
    mi = min(mi,mis)
    ma = max(ma,mas)
    
    print("min and max")
    print(mi)
    print(ma)
    
    interval = (ma-mi)/units
    imageNamesT = ['maxVal', 'totalMass']
    imageNamesD = ['maxVal','totalMass']
    for f in range(nu_T.shape[-1]):
        imageNamesT.append('zeta_' + str(f))
    for f in range(nu_D.shape[-1]):
        imageNamesD.append('zeta_' + str(f))
    if jac is not None:
        j = np.load(jac)
        j = j['j']
        imageNamesD.append('jacobian')
    
    for i in range(units):
        iT = (T[...,-1] >= mi + i*interval)*(T[...,-1] < mi + (i+1)*interval)
        nu_Ts = nu_T[iT,...]
        imageVals = [np.argmax(nu_Ts,axis=-1),np.sum(nu_Ts,axis=-1)]
        zeta_Ts = nu_Ts/np.sum(nu_Ts,axis=-1)[...,None]
        for f in range(nu_T.shape[-1]):
            imageVals.append(zeta_Ts[:,f])
        writeVTK(T[iT,...],imageVals,imageNamesT,savename+'T_zunit' + str(i) + '.vtk',polyData=None)
        iD = (D[...,-1] >= mi + i*interval)*(D[...,-1] < mi + (i+1)*interval)
        nu_Ds = nu_D[iD,...]
        imageVals = [np.argmax(nu_Ds,axis=-1),np.sum(nu_Ds,axis=-1)]
        zeta_Ds = nu_Ds/np.sum(nu_Ds,axis=-1)[...,None]
        for f in range(nu_D.shape[-1]):
            imageVals.append(zeta_Ds[:,f])
        if jac is not None:
            imageVals.append(j[iD])
        writeVTK(D[iD,...],imageVals,imageNamesD,savename+'D_zunit'+str(i) + '.vtk',polyData=None)
    return

def writeVTK(YXZ,features,featureNames,savename,polyData=None,fields=None,fieldNames=None):
    '''
    Write YXZ coordinates (assume numpts x 3 as X,Y,Z in vtk file)
    polyData should be in format 3 vertex, vertex, vertex (0 based numbering)
    '''
    f_out_data = []

    # Version 3.0 header
    f_out_data.append("# vtk DataFile Version 3.0\n")
    f_out_data.append("Surface Data\n")
    f_out_data.append("ASCII\n")
    f_out_data.append("DATASET POLYDATA\n")
    f_out_data.append("\n")

    num_pts = YXZ.shape[0]
    f_out_data.append("POINTS %d float\n" % num_pts)
    for pt in range(num_pts):
        f_out_data.append("%f %f %f\n" % (YXZ[pt,1], YXZ[pt,0], YXZ[pt,2])) # x,y,z
    
    if (polyData is not None):
        r = polyData.shape[0]
        c = polyData.shape[1]
        if (c > 3):
            f_out_data.append("POLYGONS %d %d\n" % (r,c*r))
        else:
            f_out_data.append("LINES %d %d\n" % (r,c*r))
        for i in range(r):
            if (c == 4):
                f_out_data.append("%d %d %d %d\n" % (polyData[i,0], polyData[i,1], polyData[i,2], polyData[i,3]))
            elif (c == 3):
                f_out_data.append("%d %d %d\n" % (polyData[i,0], polyData[i,1], polyData[i,2]))
    f_out_data.append("POINT_DATA %d\n" % num_pts)
    fInd = 0;
    for f in featureNames:
        f_out_data.append("SCALARS " + f + " float 1\n")
        f_out_data.append("LOOKUP_TABLE default\n")
        fCurr = features[fInd]
        for pt in range(num_pts):
            f_out_data.append("%.9f\n" % fCurr[pt])
        fInd = fInd + 1
    if fields is not None:
        for f in range(len(fields)):
            f_out_data.append("VECTORS " + fieldNames[f] + " float\n")
            fieldCurr = fields[f]
            for pt in range(num_pts):
                f_out_data.append("%f %f %f\n" % (fieldCurr[pt,1], fieldCurr[pt,0], fieldCurr[pt,2]))
                

    # Write output data array to file
    with open(savename, "w") as f_out:
        f_out.writelines(f_out_data)
    return

def writeParticleVTK(Xt,nu_Xt,savename,condense=False,featNames=None):
    if torch.is_tensor(Xt):
        X = Xt.cpu().numpy()
        nuX = nu_Xt.cpu().numpy()
    else:
        X = Xt
        nuX = nu_Xt
    
    if len(nuX.shape) < 2 or nuX.shape[-1] < 2:
        imageNames = ['Weight']
        imageVals = [np.squeeze(nuX)]
        writeVTK(X,imageVals,imageNames,savname)
    else:
        imageNames = ['Weight','Maximum_Feature_Dimension','Entropy']
        imageVals = [np.sum(nuX,axis=-1),np.argmax(nuX,axis=-1)+1]
        zetaX = nuX/np.sum(nuX,axis=-1)[...,None]
        e = np.zeros_like(zetaX)
        e[zetaX > 0] = zetaX[zetaX > 0]*np.log(zetaX[zetaX > 0])
        imageVals.append(np.sum(e,axis=-1))
        if not condense:  
            for f in range(nuX.shape[-1]):
                if featNames is not None:
                    imageNames.append(featNames[f] + '_Probabilities')
                else:
                    imageNames.append('Feature_' + str(f) + '_Probabilities')
                imageVals.append(zetaX[:,f])
        writeVTK(X,imageVals,imageNames,savename)
    return

def writeVectorField(Gs,Ge,outpath):
    '''
    Deformed grid coordinates (start and end).
    '''
    vec = np.sqrt(np.sum((Ge-Gs)**2,axis=-1))[...,None] # displacement
    print("vec size is ", vec.shape)
    polyData = np.zeros((Gs.shape[0],3))
    polyData[:,0] = 2
    polyData[:,1] = np.arange(Gs.shape[0])
    polyData[:,2] = np.arange(Gs.shape[0])+Gs.shape[0]
    
    vec0 = np.zeros_like(vec)
    vecTotal = np.vstack((vec0,vec))
    print("vec Total size is, ", vecTotal.shape)
    
    writeVTK(np.vstack((Gs,Ge)),[vecTotal],['DISPLACEMENT'],outpath,polyData=polyData)
    Gsmall = 0.1*(Ge-Gs) + Gs
    writeVTK(np.vstack((Gs,Gsmall)),[vecTotal],['DISPLACEMENT'],outpath.replace('.vtk','_small.vtk'),polyData=polyData)
    return
                   
                        
    
        
    

