# Guided Anisotropic Diffusion algorithm
# Rodrigo Caye Daudt
# https://rcdaudt.github.io
# 
# Caye Daudt, Rodrigo, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau. "Guided anisotropic diffusion and iterative learning for weakly supervised change detection." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 0-0. 2019.

import torch


def g(x, K=5):
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))


def c(I, K =5):
    cv = g(torch.unsqueeze(torch.mean(I[:,:,1:,:] - I[:,:,:-1,:], 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(I[:,:,:,1:] - I[:,:,:,:-1], 1), 1), K)
    
    return cv, ch
        
    
def diffuse_step(cv, ch, I, l=0.24):
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv
    del(dv,tv)
    
    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th
    del(dh,th)
    
    return I


def anisotropic_diffusion(I1, I2, I, N=500, l=0.24, K=5, is_log=True, verbose=False):
    if is_log:
        I = torch.exp(I)
    
    with torch.no_grad():
        for t in range(N): 
            if verbose:
                print('Iteration {}'.format(t))

            cv1, ch1 = c(I1, K=K)
            I1 = diffuse_step(cv1, ch1, I1, l=l)
            
            cv2, ch2 = c(I2, K=K)
            I2 = diffuse_step(cv2, ch2, I2, l=l)
            
            cv = torch.min(cv1, cv2)
            ch = torch.min(ch1, ch2)
            del(cv1, ch1, cv2, ch2)
            I = diffuse_step(cv, ch, I, l=l)

            del(cv,ch)
        
    return I