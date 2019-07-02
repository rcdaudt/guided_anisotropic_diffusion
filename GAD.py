# Guided Anisotropic Diffusion algorithm
# Rodrigo Caye Daudt
# https://rcdaudt.github.io
# 
# Caye Daudt, Rodrigo, Bertrand Le Saux, Alexandre Boulch, and Yann Gousseau. "Guided anisotropic diffusion and iterative learning for weakly supervised change detection." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 0-0. 2019.

import torch


def g(x, K=5):
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))


def c(I, K =5):
    cv = g(torch.mean(I[:,:,1:,1:-1] - I[:,:,:-1,1:-1], 1), K)
    ch = g(torch.mean(I[:,:,1:-1,1:] - I[:,:,1:-1,:-1], 1), K)
    
    return cv, ch
        
    
def c_pair(I1, I2, K = 5):
    cv1, ch1 = c(I1, K)
    cv2, ch2 = c(I2, K)
    
    cv = torch.min(cv1, cv2)
    ch = torch.min(ch1, ch2)
    
    return cv, ch




def anisotropic_diffusion(I1, I2, I, N=500, l=0.24, K=5, is_log=True, verbose=False):
    if is_log:
        I = torch.exp(I)

    
    with torch.no_grad():
        for t in range(N): 
            if verbose:
                print('Iteration {}'.format(t))
                   

            cv1, ch1 = c(I1, K=K)

            dv = I1[:,:,1:,1:-1] - I1[:,:,:-1,1:-1]
            dh = I1[:,:,1:-1,1:] - I1[:,:,1:-1,:-1]
            for channel in range(I1.size(1)):
                I1[:,channel,1:-1,1:-1] += l * (cv1[:,1:,:]*dv[:,channel,1:,:] - cv1[:,:-1,:]*dv[:,channel,:-1,:] + ch1[:,:,1:]*dh[:,channel,:,1:] - ch1[:,:,:-1]*dh[:,channel,:,:-1]) 
            del(dv,dh)
            
            cv2, ch2 = c(I2, K=K)

            dv = I2[:,:,1:,1:-1] - I2[:,:,:-1,1:-1]
            dh = I2[:,:,1:-1,1:] - I2[:,:,1:-1,:-1]
            for channel in range(I2.size(1)):
                I2[:,channel,1:-1,1:-1] += l * (cv2[:,1:,:]*dv[:,channel,1:,:] - cv2[:,:-1,:]*dv[:,channel,:-1,:] + ch2[:,:,1:]*dh[:,channel,:,1:] - ch2[:,:,:-1]*dh[:,channel,:,:-1]) 
            del(dv,dh)
            
            cv = torch.min(cv1, cv2)
            ch = torch.min(ch1, ch2)
            del(cv1, ch1, cv2, ch2)

            dv = I[:,:,1:,1:-1] - I[:,:,:-1,1:-1]
            dh = I[:,:,1:-1,1:] - I[:,:,1:-1,:-1]
            for channel in range(I.size(1)):
                I[:,channel,1:-1,1:-1] += l * (cv[:,1:,:]*dv[:,channel,1:,:] - cv[:,:-1,:]*dv[:,channel,:-1,:] + ch[:,:,1:]*dh[:,channel,:,1:] - ch[:,:,:-1]*dh[:,channel,:,:-1]) 
            del(dv,dh)
            
        
    return I