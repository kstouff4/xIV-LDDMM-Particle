import analyzeOutput as ao
import numpy as np


def main():
    # path to results
    fpath = "/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/BarSeq/AllenAtlasToBarSeq/output_dl_sig_its_albega_N-35_[0.2, 0.1][0.5, 0.2, 0.05]_100_0.1None_18706flipFullAtlas/"
    """
    x = np.load(fpath + 'testOutput_Dvars.npz')
    Td = x['Td']
    
    y = np.load(fpath + 'testOutput_values.npz')
    pi_ST = y['pi_ST']
    
    z = np.load(fpath + 'testOutput.npz')
    T = z['T']
    nu_T = z['nu_T']
    
    # load higher resolution atlas
    w = np.load('/cis/home/kstouff4/Documents/MeshRegistration/Particles/AllenAtlas10um/Final/Sub_13slabs_0-1_optimalZnu_ZAllwC1.2_sig0.05_Nmax5000.0_Npart1000.0.npz')
    S = w[w.files[0]]
    nu_S = w[w.files[1]]
    
    
    # flip Allen atlas over z axis
    S[:,-1] = -1.0*S[:,-1]
    _,nu_TSpi = ao.interpolateNN(Td,T,S,nu_S,pi_ST,fpath+'imageOutput/interpolateNN_onT.npz')
    
    # make images
    ao.getCompareDensity(T,nu_T,T,nu_TSpi,0.05,fpath + 'imageOutput/interpolateNN_onT',coef=1)
    
    # normalize images
    ao.removeZerosAndNormalize(fpath+'imageOutput/interpolateNN_onTTdensity.nii.gz',fpath+'imageOutput/interpolateNN_onTDdensity.nii.gz',norm=True)
    """
    ao.removeZerosAndNormalize(
        fpath + "imageOutput/interpolateNN_onTTmaxval.nii.gz",
        fpath + "imageOutput/interpolateNN_onTDmaxval.nii.gz",
        norm=False,
    )

    return


if __name__ == "__main__":
    main()
