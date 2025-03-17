import functools

import torch
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from sampling import shared_corrector_update_fn, shared_predictor_update_fn
from physics.ct import *
from utils import clear, batchfy


def grad_x(x):
    Dx1 = (torch.roll(x, shifts=1, dims=2) - x)
    Dx2 = (torch.roll(x, shifts=1, dims=3) - x)
    
    return torch.stack((Dx1, Dx2), dim=-1)
    
def grad_x_trans(x):
    y1 = (torch.roll(x[..., 0], shifts=-1, dims=2) - x[..., 0])
    y2 = (torch.roll(x[..., 1], shifts=-1, dims=3) - x[..., 1])

    return y1 + y2

def grad_v(V):
    D1v1 = (torch.roll(V[..., 0], shifts=1, dims=2) - V[..., 0])
    D2v2 = (torch.roll(V[..., 1], shifts=1, dims=3) - V[..., 1])
    D2v1 = (torch.roll(V[..., 0], shifts=1, dims=3) - V[..., 0])
    D1v2 = (torch.roll(V[..., 1], shifts=1, dims=2) - V[..., 1])
    
    return torch.stack((D1v1, D2v2, (D2v1 + D1v2) / 2), dim=-1)
  
def grad_v_trans(V):
    y1 = (torch.roll(V[..., 0], shifts=-1, dims=2) - V[..., 0]) + (torch.roll(V[..., 2], shifts=-1, dims=3) - V[..., 2]) / 2
    y2 = (torch.roll(V[..., 1], shifts=-1, dims=3) - V[..., 1]) + (torch.roll(V[..., 2], shifts=-1, dims=2) - V[..., 2]) / 2

    return torch.stack((y1, y2), dim=4)



def get_pc_radon_ADMM_TGV_vol(sde, predictor, corrector, inverse_scaler, snr,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                             final_consistency=False, img_shape=None, lam=0.1,
                             rho_0=10, rho_1=10, alpha_0=1, alpha_1=1):
    """ Sparse application of measurement consistency """
    # Define predictor & corrector
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    
    x_0 : Tensor = torch.zeros(img_shape)
    Dx = grad_x(x_0)
    del_v : Tensor = Dx
    del_z : Tensor = Dx - del_v
    Dv = grad_v(del_v)
    del_y : Tensor = Dv
    del_Uz : Tensor = del_z - Dx + del_v
    del_Uy : Tensor = Dv - del_y
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lam=1.0, i=None, norm_const=None):
        x = x + lam * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean


    def Z_step(Dx, del_v, del_Uz, del_z, lam=0.1, alpha_1=1, rho_0=10):
        """
        Optimized proximal gradient for l2 norm
        This function is modified to handle inputs with shape (448, 1, 256, 256, 2)
        """
        
        assert Dx.shape == del_v.shape == del_Uz.shape == del_z.shape
        
        original_shape = Dx.shape
        Dx_flat = Dx.reshape(-1, 256*256*2)
        del_v_flat = del_v.reshape(-1, 256*256*2)
        del_Uz_flat = del_Uz.reshape(-1, 256*256*2)
        
        a = (Dx_flat - del_v_flat - del_Uz_flat).reshape(-1, 2)
        coef = torch.maximum(1 - (2 * lam * alpha_1 / rho_0) / torch.norm(a, dim=1, keepdim=True), torch.tensor(0.0))
        result_flat = (coef * a).reshape(-1, 256*256*2)
        
        result = result_flat.reshape(original_shape)
        return result
    
       

    def Y_step(DV, del_Uy, del_y, lam=0.1, alpha_0=1, rho_1=10):
        """
        Optimized proximal gradient for l2 norm
        Modified to handle inputs with shape (448, 1, 256, 256, 3)
        """
        
        assert DV.shape == del_Uy.shape == del_y.shape
        
        original_shape = DV.shape
        DV_flat = DV.reshape(-1, 256*256*3)
        del_Uy_flat = del_Uy.reshape(-1, 256*256*3)
        
        a = (DV_flat + del_Uy_flat).reshape(-1, 3)
        coef = torch.maximum(1 - (2 * lam * alpha_0 / rho_1) / torch.norm(a, dim=1, keepdim=True), torch.tensor(0.0))
        result_flat = (coef * a).reshape(-1, 256*256*3)
        
        result = result_flat.reshape(original_shape)
        return result


    def A_cg1(x):
        return _AT(_A(x)) + 2 * rho_0 * grad_x_trans(grad_x(x))

    def A_cg2(v):
        return 2 * rho_0 * v + 2 * rho_1 * grad_v_trans(grad_v(v))



    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)
        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)
            x += a * p
            r -= a * Ap
            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x



    def CS_routine(x : Tensor, ATy, niter):
        nonlocal del_z, del_v, del_y, del_Uz, del_Uy
        if del_z.device != x.device:
            del_z = del_z.to(x.device)
            del_v = del_v.to(x.device)
            del_y = del_y.to(x.device)
            del_Uz = del_Uz.to(x.device)
            del_Uy = del_Uy.to(x.device)
        for i in range(niter):
            b_cg1 = ATy + 2 * rho_0 * grad_x_trans(del_z + del_v + del_Uz)
            x = CG(A_cg1, b_cg1, x, n_inner=1)
            EtYU = grad_v_trans(del_y - del_Uy)
            b_cg2 = 2 * rho_0 * (grad_x(x) - del_z - del_Uz) + 2 * rho_1 * EtYU
            del_v = CG(A_cg2, b_cg2, del_v, n_inner=1)
            del_z = Z_step(grad_x(x),del_v,del_Uz, del_z, lam=lam, alpha_1=alpha_1, rho_0=rho_0)
            del_y = Y_step(grad_v(del_v), del_Uy, del_y, lam=lam, alpha_0=alpha_0, rho_1=rho_1)
            del_Uz += del_z - grad_x(x) + del_v
            del_Uy += grad_v(del_v) - del_y 
        x_mean = x
        return x, x_mean
    
    
    

    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean

        return radon_update_fn



    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean
        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()


    def pc_radon(model, data, measurement=None, batch_size=12):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)
            
            ones = torch.ones_like(x).to(data.device) 
            norm_const = _AT(_A(ones)) 
            timesteps = torch.linspace(sde.T, eps, sde.N) 
            for i in tqdm(range(sde.N)): 
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU 
                x_batch = batchfy(x, batch_size)
                # 2. Run PC step for each batch
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch): 
                    x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_sing, t)
                    x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)
                if save_progress:
                    if (i % 20) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[0:1]), cmap='gray')
            # Final step which coerces the data fidelity error term to be zero,
            # and thereby satisfying Ax = y
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lam=1.0, norm_const=norm_const)
            return inverse_scaler(x_mean if denoise else x)
    return pc_radon