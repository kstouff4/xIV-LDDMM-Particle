import torch

from xmodmap.distance.LossVarifoldNormBoundary import LossVarifoldNormBoundary
#from xmodmap.distance.LossVarifoldNorm import GaussLinKernelSingle


class LossVarifoldNormBoundary_Partial(LossVarifoldNormBoundary):
    '''
    Support weights reflect density of particles in target space and interior of set of coronal sections.
    
    densSigma = bandwith (for data in unit cube) of Gaussian kernel used to compute local density of particles
    in target space.
    '''

    def __init__(self, sigmaVar, w_T, zeta_T, Ttilde, densSigma):
        
        super().__init__(sigmaVar, w_T, zeta_T, Ttilde)
        self.densSigma = densSigma
        self.Kd = self.GaussLinKernelSingle(self.densSigma,self.Ttilde.shape[-1],1)
        
        if self.w_T.shape[-1] != 1:
            self.w_T = self.w_T[...,None]
        
    
    def densityWeight(self,qx):
        '''
        Assigns weight at local points (qx) based on density of particles in target space.
        Density computed with local Gaussian smoothing, with bandwith densSigma.
        '''
        
        # TO DO: normalize weights?
        weights = self.Kd(qx,self.Ttilde,torch.ones((qx.shape[0],1)),self.w_T)
        return weights
    
    def supportWeight(self, qx, lamb):

        sW = (0.5 * torch.tanh(torch.sum((qx - self.a0) * self.n0, axis=-1) / lamb)
                + 0.5 * torch.tanh(torch.sum((qx - self.a1) * self.n1, axis=-1) / lamb))[..., None]
        return self.densityWeight(qx)*sW
    
    def __call__(self, sSx, sSw, zeta_S, pi_ST, lamb):
        return super().__call__(sSx, sSw, zeta_S, pi_ST, lamb)

