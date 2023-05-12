import sys
from sys import path as sys_path
sys_path.append('../xmodmap/io/')
import getOutput as gO

def main():
    pref = '/cis/home/kstouff4/Documents/MeshRegistration/ParticleLDDMMQP/sandbox/AllenMerfish/AllenAtlasToMERFISH/output_dl_sig_its_albega_N-320_[0.2, 0.1][0.5, 0.2, 0.05]_100_1.0None_50477FullSet/'
    origST = pref + 'origST.npz'
    dVars = pref + 'testOutput_Dvars.npz'
    
    iD = np.load(dVars)
    D = iD['D']
    nu_D = iD['nu_D']
    nu_Dpi = iD['nu_Dpi']
    
    iO = np.load(origST)
    S = iO['S']
    nu_S = iO['nu_S']
    T = iO['T']
    nu_T = iO['nu_T']
    
    jFile = gO.getJacobian(D,nu_S,nu_D,pref+'testOutput_D10_jac.vtk')
    gO.splitZs(T,nu_T,D,nu_Dpi,pref+'out',units=10,jac=jFile)
    
    return

if __name__ == "__main__":
    main()