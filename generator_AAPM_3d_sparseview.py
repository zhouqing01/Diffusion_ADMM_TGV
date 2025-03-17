import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import numpy as np
import controllable_generation_TV
import controllable_generation_TGV_1
import controllable_generation_TGV_2
import controllable_generation_TGV_3

from utils import restore_checkpoint, clear, batchfy, patient_wise_min_max, img_wise_min_max
from pathlib import Path
from models import utils as mutils
from models import ncsnpp
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import datasets

# for radon
from physics.ct import CT
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import argparse


def run(p_metho='TGV', p_n_view=8, p_rho_0=10, p_rho_1=10, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device='', batch_size=12):
    ###############################################
    # Configurations
    ###############################################
    problem = 'sparseview_CT_ADMM_TV_total'
    config_name = 'AAPM_256_ncsnpp_continuous'
    sde = 'VESDE'
    num_scales = 2000
    ckpt_num = 185
    N = num_scales

    vol_name = 'L067'
    root = Path(f'./data/CT/ind/256_sorted/{vol_name}')

    # Parameters for the inverse problem
    metho = p_metho
    n_view = p_n_view
    det_spacing = 1.0
    size = 256
    det_count = int((size * (2 * torch.ones(1)).sqrt()).ceil())
    lam = p_lam
    rho_0 = p_rho_0
    rho_1 = p_rho_1
    alpha_0 = p_alpha_0
    alpha_1 = p_alpha_1
    freq = 1

    if sde.lower() == 'vesde':
        from configs.ve import AAPM_256_ncsnpp_continuous as configs
        ckpt_filename = f"exp/ve/{config_name}/checkpoint_{ckpt_num}.pth"
        config = configs.get_config()
        config.model.num_scales = N
        sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
        sde.N = N
        sampling_eps = 1e-5
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16
    n_steps = 1

    if device != '':
        if device == '0':
            config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        elif device == '1':
            config.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    batch_size = 12
    config.training.batch_size = batch_size
    config.eval.batch_size = batch_size
    random_seed = 0

    sigmas = mutils.get_sigmas(config)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    score_model = mutils.create_model(config, device=config.device)  ## model

    optimizer = get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, optimizer=optimizer,
                 model=score_model, ema=ema)
    
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    if metho == 'TV':
        save_root = Path(f'./results/{config_name}/{problem}/{metho}/n_view{n_view}/rho_{rho_0}-lambda_{lam}')
    else:
        save_root = Path(f'./results/{config_name}/{problem}/{metho}/n_view{n_view}/rho0_{rho_0}-rho1_{rho_1}-lambda_{lam}-alpha0_{alpha_0}-alpha_1_{alpha_1}')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'label', 'BP', 'sinogram', 'volume']
    for t in irl_types:
        if t == 'recon':
            save_root_f = save_root / t / 'progress'
            save_root_f.mkdir(exist_ok=True, parents=True)
        else:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)

    # read all data
    fname_list = os.listdir(root)
    fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
    print(fname_list)
    all_img = []

    print("Loading all data")
    for fname in tqdm(fname_list):
        just_name = fname.split('.')[0]
        img = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))
        h, w = img.shape
        img = img.view(1, 1, h, w)
        all_img.append(img)
        plt.imsave(os.path.join(save_root, 'label', f'{just_name}.png'), clear(img), cmap='gray')
    all_img = torch.cat(all_img, dim=0)
    print(f"Data loaded shape : {all_img.shape}")

    # full
    angles = np.linspace(0, np.pi, 180, endpoint=False)
    radon = CT(img_width=h, radon_view=n_view, uniform=True, circle=False, device=config.device)

    predicted_sinogram = []
    label_sinogram = []
    img_cache = None

    img = all_img.to(config.device)

    if metho == 'TV':
        pc_radon = controllable_generation_TV.get_pc_radon_ADMM_TV_vol(sde,
                                                                    predictor, corrector,
                                                                    inverse_scaler,
                                                                    snr=snr,
                                                                    n_steps=n_steps,
                                                                    probability_flow=probability_flow,
                                                                    continuous=config.training.continuous,
                                                                    denoise=True,
                                                                    radon=radon,
                                                                    save_progress=True,
                                                                    save_root=save_root,
                                                                    final_consistency=True,
                                                                    img_shape=img.shape,
                                                                    lamb_1=lam,
                                                                    rho=rho_0)
        
    elif metho == 'TGV_3':
        pc_radon = controllable_generation_TGV_3.get_pc_radon_ADMM_TGV_vol(sde,
                                                                        predictor, corrector,
                                                                        inverse_scaler,
                                                                        snr=snr,
                                                                        n_steps=n_steps,
                                                                        probability_flow=probability_flow,
                                                                        continuous=config.training.continuous,
                                                                        denoise=True,
                                                                        radon=radon,
                                                                        save_progress=True,
                                                                        save_root=save_root,
                                                                        final_consistency=True,
                                                                        img_shape=img.shape,
                                                                        lam=lam,
                                                                        rho_0=rho_0,
                                                                        rho_1=rho_1,
                                                                        alpha_0=alpha_0,
                                                                        alpha_1=alpha_1)

    elif metho == 'TGV_2':
        pc_radon = controllable_generation_TGV_2.get_pc_radon_ADMM_TGV_vol(sde,
                                                                        predictor, corrector,
                                                                        inverse_scaler,
                                                                        snr=snr,
                                                                        n_steps=n_steps,
                                                                        probability_flow=probability_flow,
                                                                        continuous=config.training.continuous,
                                                                        denoise=True,
                                                                        radon=radon,
                                                                        save_progress=True,
                                                                        save_root=save_root,
                                                                        final_consistency=True,
                                                                        img_shape=img.shape,
                                                                        lam=lam,
                                                                        rho_0=rho_0,
                                                                        rho_1=rho_1,
                                                                        alpha_0=alpha_0,
                                                                        alpha_1=alpha_1)
        
    elif metho == 'TGV_1':
        pc_radon = controllable_generation_TGV_1.get_pc_radon_ADMM_TGV_vol(sde,
                                                                        predictor, corrector,
                                                                        inverse_scaler,
                                                                        snr=snr,
                                                                        n_steps=n_steps,
                                                                        probability_flow=probability_flow,
                                                                        continuous=config.training.continuous,
                                                                        denoise=True,
                                                                        radon=radon,
                                                                        save_progress=True,
                                                                        save_root=save_root,
                                                                        final_consistency=True,
                                                                        img_shape=img.shape,
                                                                        lam=lam,
                                                                        rho_0=rho_0,
                                                                        rho_1=rho_1,
                                                                        alpha_0=alpha_0,
                                                                        alpha_1=alpha_1)
        
    else:
        raise ValueError(f"Invalid method: {metho}")
    
    sinogram = radon.A(img.to(config.device))

    # A_dagger
    bp = radon.AT(sinogram)

    # Recon Image
    #x = pc_radon(score_model, scaler(img), measurement=sinogram)
    x = pc_radon(score_model, scaler(img.to(config.device)), measurement=sinogram.to(config.device), batch_size=batch_size) 

    img_cahce = x[-1].unsqueeze(0)

    count = 0
    for i, recon_img in enumerate(x):
        plt.imsave(save_root / 'BP' / f'{count}.png', clear(bp[i]), cmap='gray')
        plt.imsave(save_root / 'label' / f'{count}.png', clear(img[i]), cmap='gray')
        plt.imsave(save_root / 'recon' / f'{count}.png', clear(recon_img), cmap='gray')
        count += 1
        print("the count number is:",count)

    # Recon and Save Sinogram
    #label_sinogram.append(radon.A_all(img))
    #predicted_sinogram.append(radon.A_all(x))
    label_sinogram.append(radon.A_all(img.to(config.device)))
    predicted_sinogram.append(radon.A_all(x.to(config.device)))

    original_sinogram = torch.cat(label_sinogram, dim=0).detach().cpu().numpy()
    recon_sinogram = torch.cat(predicted_sinogram, dim=0).detach().cpu().numpy()

    try:
        np.save(str(save_root / 'sinogram' / f'original_{count}.npy'), original_sinogram)
        np.save(str(save_root / 'sinogram' / f'recon_{count}.npy'), recon_sinogram)
        np.save(str(save_root / 'volume' / f'volume_{count}.npy'), clear(x))
        np.save(str(save_root / 'volume' / f'ground_truth_{count}.npy'), clear(img))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparseview CT ADMM TV Total Variation')
    
    parser.add_argument('--metho', type=str, default='TGV', help='Method to use: TGV or TV')
    parser.add_argument('--n_view', type=int, default=8, help='Number of views')
    parser.add_argument('--rho_0', type=int, default=10, help='Rho 0')
    parser.add_argument('--rho_1', type=int, default=10, help='Rho 1')
    parser.add_argument('--alpha_0', type=int, default=1, help='Alpha 0')
    parser.add_argument('--alpha_1', type=int, default=1, help='Alpha 1')
    parser.add_argument('--lam', type=float, default=0.04, help='Lambda')
    args = parser.parse_args()

    run(args.metho, args.n_view, args.rho_0, args.rho_1, args.alpha_0, args.alpha_1, args.lam)
