import functools
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
from sampling import shared_corrector_update_fn, shared_predictor_update_fn
from physics.ct import *
from utils import clear, batchfy


def grad_x(x : Tensor):
    return torch.roll(x, shifts=-1, dims=0) - x

def grad_x_trans(x : Tensor):
    return torch.roll(x, shifts=1, dims=0) - x

def grad_g(g : Tensor):
    return torch.roll(g, shifts=-1, dims=0) - g

def grad_g_trans(g : Tensor):
    return torch.roll(g, shifts=1, dims=0) - g


def get_pc_radon_ADMM_TGV_vol(sde, predictor, corrector, inverse_scaler, snr,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, radon=None, save_progress=False, save_root=None,
                             final_consistency=False, img_shape=None, 
                             lam=0.1, rho_0=10, rho_1=10, alpha_0=1, alpha_1=1):
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
    
    
    g : Tensor = torch.zeros(img_shape)
    z : Tensor = torch.zeros(img_shape)
    y : Tensor = torch.zeros(img_shape)
    u_z : Tensor = torch.zeros(img_shape)
    u_y : Tensor = torch.zeros(img_shape)
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)
    
    def kaczmarz(x, x_mean, measurement=None, lam=1.0, i=None, norm_const=None):
        x = x + lam * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean
    
    def z_step(dx : Tensor, v : Tensor, z : Tensor, u_z : Tensor, lam=0.1, alpha_1=1, rho_0=10):
        """
        Optimized proximal gradient for l2 norm
        This function is modified to handle inputs with shape (448, 1, 256, 256, 3)
        """
        # check shape
        assert dx.shape == v.shape == u_z.shape == z.shape
        # flat
        original_shape = dx.shape
        Dx_flat = dx.reshape(-1, 256*256*1)
        del_v_flat = v.reshape(-1, 256*256*1)
        del_Uz_flat = u_z.reshape(-1, 256*256*1)
        # compute
        a = (Dx_flat - del_v_flat - del_Uz_flat).reshape(-1, 1)
        coef = torch.maximum(1 - (2 * lam * alpha_1 / rho_0) / torch.norm(a, dim=1, keepdim=True), torch.tensor(0.0))
        result_flat = (coef * a).reshape(-1, 256*256*1)
        # reshape
        result = result_flat.reshape(original_shape)
        return result
    
    def y_setp(dg : Tensor, y : Tensor, u_y : Tensor, lam=0.1, alpha_0=1, rho_1=10):
        """
        Optimized proximal gradient for l2 norm
        This function is modified to handle inputs with shape (448, 1, 256, 256, 6)
        """
        # check shape
        assert dg.shape == u_y.shape == y.shape
        # flat
        original_shape = dg.shape
        dg_flat = dg.reshape(-1, 256*256*1)
        del_u_y_flat = u_y.reshape(-1, 256*256*1)
        # compute
        a = (dg_flat + del_u_y_flat).reshape(-1, 1)
        coef = torch.maximum(1 - (2 * lam * alpha_0 / rho_1) / torch.norm(a, dim=1, keepdim=True), torch.tensor(0.0))
        result_flat = (coef * a).reshape(-1, 256*256*1)
        # reshape
        result = result_flat.reshape(original_shape)
        return result
    
    def A_cg_x(x):
        return _AT(_A(x)) + 2 * rho_0 * grad_x_trans(grad_x(x))

    def A_cg_g(g):
        return 2 * rho_0 * g + 2 * rho_1 * grad_g_trans(grad_g(g))
    
    def CG(A_fn, b_cg : Tensor, x : Tensor, n_inner=10):
        r : Tensor = b_cg - A_fn(x)
        p : Tensor = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)
        for i in range(n_inner):
            Ap : Tensor = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)
            x += a * p
            r -= a * Ap
            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x
    
    def CS_routine(x : Tensor, ATb, niter):
        nonlocal g, z, y, u_z, u_y
        if g.device != x.device:
            g = g.to(x.device)
            z = z.to(x.device)
            y = y.to(x.device)
            u_z = u_z.to(x.device)
            u_y = u_y.to(x.device)

        for i in range(niter):
            # x step
            b_cg_x = ATb + 2 * rho_0 * grad_x_trans(z + g + u_z)
            x = CG(A_cg_x, b_cg_x, x, n_inner=1)

            # g step
            EtYU = grad_g_trans(y - u_y)
            b_cg_g = 2 * rho_0 * (grad_x(x) - z - u_z) + 2 * rho_1 * EtYU
            g = CG(A_cg_g, b_cg_g, g, n_inner=1)

            # z step
            z = z_step(grad_x(x), g, z, u_z, lam=lam, alpha_1=alpha_1, rho_0=rho_0)

            # y step
            y = y_setp(grad_g(g), y, u_y, lam=lam, alpha_0=alpha_0, rho_1=rho_1)

            # update u_z and u_y    
            u_z += z - grad_x(x) + g
            u_y += grad_g(g) - y

        x_mean = x
        return x, x_mean
    
    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x : Tensor, t):
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
