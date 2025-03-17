import argparse

import generator_AAPM_3d_sparseview
import generator_AAPM_3d_limitedangle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experience.')

    parser.add_argument('--view', type=str, default='8,4,2,90', help='Number of views')
    parser.add_argument('--method', type=str, default='', help='Method')
    parser.add_argument('--device', type=str, default='', help='Device')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
    args = parser.parse_args()

    views = args.view.split(',')
    methods = args.method.split(',')

    if '8' in views:
        if 'TGV_3' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_3', p_n_view=8, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TGV_2' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_2', p_n_view=8, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TGV_1' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_1', p_n_view=8, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=8, p_rho_0=10, p_lam=0.04, device=args.device, batch_size=args.batch_size)

    if '4' in views:
        if 'TGV_3' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_3', p_n_view=4, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TGV_2' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_2', p_n_view=4, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TGV_1' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_1', p_n_view=4, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=4, p_rho_0=10, p_lam=0.04, device=args.device, batch_size=args.batch_size)

    if '2' in views:
        if 'TGV_3' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_3', p_n_view=2, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TGV_2' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_2', p_n_view=2, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TGV_1' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TGV_1', p_n_view=2, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TV' in methods or args.method == '':
            generator_AAPM_3d_sparseview.run(p_metho='TV', p_n_view=2, p_rho_0=10, p_lam=0.04, device=args.device, batch_size=args.batch_size)

    if '90' in views:
        if 'TGV_3' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_limitedangle.run(p_metho='TGV_3', p_n_view=90, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TGV_2' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_limitedangle.run(p_metho='TGV_2', p_n_view=90, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TGV_1' in methods or 'TGV' in methods or args.method == '':
            generator_AAPM_3d_limitedangle.run(p_metho='TGV_1', p_n_view=90, p_rho_0=5, p_rho_1=5, p_alpha_0=1, p_alpha_1=1, p_lam=0.04, device=args.device, batch_size=args.batch_size)
        if 'TV' in methods or args.method == '':
            generator_AAPM_3d_limitedangle.run(p_metho='TV', p_n_view=90, p_rho_0=10, p_lam=0.04, device=args.device, batch_size=args.batch_size)
