import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','ieee']) # science plot

# 数据定义
methods = ['ADMM-TV', 'FBPConvNet', 'Lahiri et al.', 'Chung et al.', 'TGV_z', 'TGV_xy', 'TGV_xyz', 'TV_z']


view_labels = ['Axial Slice', 'Sagittal Slice', 'Coronal Slice']
slices = [2, 4, 8]  # 视角数
metrics = ['PSNR', 'SSIM']

colors = [
    # '#1f77b4',  # Method 1
    # '#ff7f0e',  # Method 2
    '#2ca02c',  # Method 3
    '#d62728',  # Method 4
    '#9467bd',  # Method 5
    '#8c564b',  # Method 6
    '#e377c2',  # Method 7
    #'#7f7f7f',  # Method 8
    '#bcbd22',  # Method 9
    '#17becf',  # Method 10
    '#f4a261'   # Method 11
]

data1 = {
    'ADMM-TV': {
        2: {'Axial': {'PSNR': 10.28, 'SSIM': 0.409},
            'Coronal': {'PSNR': 13.77, 'SSIM': 0.616},
            'Sagittal': {'PSNR': 11.49, 'SSIM': 0.553}},
        4: {'Axial': {'PSNR': 13.59, 'SSIM': 0.618},
            'Coronal': {'PSNR': 15.23, 'SSIM': 0.682},
            'Sagittal': {'PSNR': 14.60, 'SSIM': 0.638}},
        8: {'Axial': {'PSNR': 16.79, 'SSIM': 0.645},
            'Coronal': {'PSNR': 18.95, 'SSIM': 0.772},
            'Sagittal': {'PSNR': 17.27, 'SSIM': 0.716}},
    },
    'FBPConvNet': {
        2: {'Axial': {'PSNR': 16.31, 'SSIM': 0.521},
            'Coronal': {'PSNR': 17.05, 'SSIM': 0.521},
            'Sagittal': {'PSNR': 11.07, 'SSIM': 0.483}},
        4: {'Axial': {'PSNR': 16.45, 'SSIM': 0.529},
            'Coronal': {'PSNR': 19.47, 'SSIM': 0.713},
            'Sagittal': {'PSNR': 15.48, 'SSIM': 0.610}},
        8: {'Axial': {'PSNR': 16.57, 'SSIM': 0.553},
            'Coronal': {'PSNR': 19.12, 'SSIM': 0.774},
            'Sagittal': {'PSNR': 18.11, 'SSIM': 0.714}},
    },
    'Lahiri et al.': {
        2: {'Axial': {'PSNR': 19.74, 'SSIM': 0.631},
            'Coronal': {'PSNR': 19.92, 'SSIM': 0.720},
            'Sagittal': {'PSNR': 17.34, 'SSIM': 0.650}},
        4: {'Axial': {'PSNR': 20.37, 'SSIM': 0.652},
            'Coronal': {'PSNR': 21.41, 'SSIM': 0.721},
            'Sagittal': {'PSNR': 18.40, 'SSIM': 0.665}},
        8: {'Axial': {'PSNR': 21.38, 'SSIM': 0.711},
            'Coronal': {'PSNR': 23.89, 'SSIM': 0.769},
            'Sagittal': {'PSNR': 20.81, 'SSIM': 0.716}},
    },
    'Chung et al.': {
        2: {'Axial': {'PSNR': 24.69, 'SSIM': 0.821},
            'Coronal': {'PSNR': 23.52, 'SSIM': 0.806},
            'Sagittal': {'PSNR': 20.71, 'SSIM': 0.685}},
        4: {'Axial': {'PSNR': 27.33, 'SSIM': 0.855},
            'Coronal': {'PSNR': 26.52, 'SSIM': 0.863},
            'Sagittal': {'PSNR': 23.04, 'SSIM': 0.745}},
        8: {'Axial': {'PSNR': 28.61, 'SSIM': 0.873},
            'Coronal': {'PSNR': 28.05, 'SSIM': 0.884},
            'Sagittal': {'PSNR': 24.45, 'SSIM': 0.765}},
    },
    'TGV_z': {
        2: {'Axial': {'PSNR': 23.98, 'SSIM': 0.673},
            'Coronal': {'PSNR': 24.08, 'SSIM': 0.653},
            'Sagittal': {'PSNR': 23.99, 'SSIM': 0.664}},
        4: {'Axial': {'PSNR': 24.23, 'SSIM': 0.738},
            'Coronal': {'PSNR': 24.42, 'SSIM': 0.722},
            'Sagittal': {'PSNR': 24.26, 'SSIM': 0.731}},
        8: {'Axial': {'PSNR': 22.10, 'SSIM': 0.691},
            'Coronal': {'PSNR': 22.96, 'SSIM': 0.672},
            'Sagittal': {'PSNR': 22.35, 'SSIM': 0.685}},
    },
    'TGV_xy': {
        2: {'Axial': {'PSNR': 22.47, 'SSIM': 0.611},
            'Coronal': {'PSNR': 22.48, 'SSIM': 0.582},
            'Sagittal': {'PSNR': 22.45, 'SSIM': 0.591}},
        4: {'Axial': {'PSNR': 26.43, 'SSIM': 0.802},
            'Coronal': {'PSNR': 26.32, 'SSIM': 0.712},
            'Sagittal': {'PSNR': 27.43, 'SSIM': 0.723}},
        8: {'Axial': {'PSNR': 31.57, 'SSIM': 0.858},
            'Coronal': {'PSNR': 31.87, 'SSIM': 0.815},
            'Sagittal': {'PSNR': 31.44, 'SSIM': 0.825}},
    },
    'TGV_xyz': {
        2: {'Axial': {'PSNR': 22.81, 'SSIM': 0.648},
            'Coronal': {'PSNR': 22.81, 'SSIM': 0.634},
            'Sagittal': {'PSNR': 22.80, 'SSIM': 0.640}},
        4: {'Axial': {'PSNR': 25.44, 'SSIM': 0.720},
            'Coronal': {'PSNR': 25.63, 'SSIM': 0.710},
            'Sagittal': {'PSNR': 25.40, 'SSIM': 0.716}},
        8: {'Axial': {'PSNR': 30.05, 'SSIM': 0.857},
            'Coronal': {'PSNR': 31.09, 'SSIM': 0.856},
            'Sagittal': {'PSNR': 30.49, 'SSIM': 0.859}},
    },
    'TV_z': {
        2: {'Axial': {'PSNR': 25.46, 'SSIM': 0.809},
            'Coronal': {'PSNR': 26.00, 'SSIM': 0.799},
            'Sagittal': {'PSNR': 25.96, 'SSIM': 0.803}},
        4: {'Axial': {'PSNR': 30.60, 'SSIM': 0.855},
            'Coronal': {'PSNR': 31.40, 'SSIM': 0.816},
            'Sagittal': {'PSNR': 30.46, 'SSIM': 0.824}},
        8: {'Axial': {'PSNR': 32.85, 'SSIM': 0.882},
            'Coronal': {'PSNR': 33.71, 'SSIM': 0.855},
            'Sagittal': {'PSNR': 32.80, 'SSIM': 0.863}},
    }
}

data2 = {
    'ADMM-TV': {
        2: {'Axial': {'PSNR': 11.49, 'SSIM': 0.568},
            'Coronal': {'PSNR': 13.86, 'SSIM': 0.619},
            'Sagittal': {'PSNR': 12.49, 'SSIM': 0.573}},
        4: {'Axial': {'PSNR': 13.89, 'SSIM': 0.638},
            'Coronal': {'PSNR': 15.37, 'SSIM': 0.686},
            'Sagittal': {'PSNR': 14.86, 'SSIM': 0.693}},
        8: {'Axial': {'PSNR': 17.22, 'SSIM': 0.674},
            'Coronal': {'PSNR': 18.27, 'SSIM': 0.773},
            'Sagittal': {'PSNR': 17.69, 'SSIM': 0.723}},
    },
    'FBPConvNet': {
        2: {'Axial': {'PSNR': 16.53, 'SSIM': 0.573},
            'Coronal': {'PSNR': 17.40, 'SSIM': 0.566},
            'Sagittal': {'PSNR': 15.07, 'SSIM': 0.435}},
        4: {'Axial': {'PSNR': 18.45, 'SSIM': 0.674},
            'Coronal': {'PSNR': 20.01, 'SSIM': 0.713},
            'Sagittal': {'PSNR': 18.45, 'SSIM': 0.710}},
        8: {'Axial': {'PSNR': 18.73, 'SSIM': 0.680},
            'Coronal': {'PSNR': 19.67, 'SSIM': 0.774},
            'Sagittal': {'PSNR': 18.30, 'SSIM': 0.741}},
    },
    'Lahiri et al.': {
        2: {'Axial': {'PSNR': 20.31, 'SSIM': 0.621},
            'Coronal': {'PSNR': 21.09, 'SSIM': 0.637},
            'Sagittal': {'PSNR': 20.07, 'SSIM': 0.583}},
        4: {'Axial': {'PSNR': 22.49, 'SSIM': 0.657},
            'Coronal': {'PSNR': 23.18, 'SSIM': 0.742},
            'Sagittal': {'PSNR': 21.94, 'SSIM': 0.668}},
        8: {'Axial': {'PSNR': 24.95, 'SSIM': 0.738},
            'Coronal': {'PSNR': 25.12, 'SSIM': 0.778},
            'Sagittal': {'PSNR': 24.65, 'SSIM': 0.768}},
    },
    'Chung et al.': {
        2: {'Axial': {'PSNR': 24.69, 'SSIM': 0.783},
            'Coronal': {'PSNR': 24.52, 'SSIM': 0.806},
            'Sagittal': {'PSNR': 23.73, 'SSIM': 0.712}},
        4: {'Axial': {'PSNR': 27.76, 'SSIM': 0.827},
            'Coronal': {'PSNR': 27.53, 'SSIM': 0.836},
            'Sagittal': {'PSNR': 26.04, 'SSIM': 0.803}},
        8: {'Axial': {'PSNR': 28.63, 'SSIM': 0.857},
            'Coronal': {'PSNR': 28.29, 'SSIM': 0.841},
            'Sagittal': {'PSNR': 27.02, 'SSIM': 0.819}},
    },
    'TGV_z': {
        2: {'Axial': {'PSNR': 26.26, 'SSIM': 0.773},
            'Coronal': {'PSNR': 26.70, 'SSIM': 0.769},
            'Sagittal': {'PSNR': 25.39, 'SSIM': 0.769}},
        4: {'Axial': {'PSNR': 28.26, 'SSIM': 0.799},
            'Coronal': {'PSNR': 29.15, 'SSIM': 0.809},
            'Sagittal': {'PSNR': 28.33, 'SSIM': 0.810}},
        8: {'Axial': {'PSNR': 27.97, 'SSIM': 0.801},
            'Coronal': {'PSNR': 28.11, 'SSIM': 0.827},
            'Sagittal': {'PSNR': 27.95, 'SSIM': 0.831}},
    },
    'TGV_xy': {
        2: {'Axial': {'PSNR': 22.86, 'SSIM': 0.709},
            'Coronal': {'PSNR': 24.29, 'SSIM': 0.702},
            'Sagittal': {'PSNR': 22.98, 'SSIM': 0.700}},
        4: {'Axial': {'PSNR': 25.50, 'SSIM': 0.680},
            'Coronal': {'PSNR': 25.96, 'SSIM': 0.690},
            'Sagittal': {'PSNR': 25.52, 'SSIM': 0.692}},
        8: {'Axial': {'PSNR': 28.48, 'SSIM': 0.839},
            'Coronal': {'PSNR': 29.22, 'SSIM': 0.851},
            'Sagittal': {'PSNR': 28.61, 'SSIM': 0.852}},
    },
    'TGV_xyz': {
        2: {'Axial': {'PSNR': 25.13, 'SSIM': 0.775},
            'Coronal': {'PSNR': 26.56, 'SSIM': 0.768},
            'Sagittal': {'PSNR': 25.25, 'SSIM': 0.767}},
        4: {'Axial': {'PSNR': 28.13, 'SSIM': 0.812},
            'Coronal': {'PSNR': 29.27, 'SSIM': 0.814},
            'Sagittal': {'PSNR': 28.35, 'SSIM': 0.815}},
        8: {'Axial': {'PSNR': 29.13, 'SSIM': 0.839},
            'Coronal': {'PSNR': 30.13, 'SSIM': 0.847},
            'Sagittal': {'PSNR': 29.34, 'SSIM': 0.849}},
    },
    'TV_z': {
        2: {'Axial': {'PSNR': 26.01, 'SSIM': 0.780},
            'Coronal': {'PSNR': 26.35, 'SSIM': 0.778},
            'Sagittal': {'PSNR': 25.07, 'SSIM': 0.777}},
        4: {'Axial': {'PSNR': 30.60, 'SSIM': 0.838},
            'Coronal': {'PSNR': 31.67, 'SSIM': 0.845},
            'Sagittal': {'PSNR': 30.63, 'SSIM': 0.846}},
        8: {'Axial': {'PSNR': 31.23, 'SSIM': 0.875},
            'Coronal': {'PSNR': 31.77, 'SSIM': 0.887},
            'Sagittal': {'PSNR': 31.27, 'SSIM': 0.885}},
    }
}


data3 = {
    'ADMM-TV': {
        2: {'Axial': {'PSNR': 10.75, 'SSIM': 0.415},
            'Coronal': {'PSNR': 13.72, 'SSIM': 0.647},
            'Sagittal': {'PSNR': 12.96, 'SSIM': 0.569}},
        4: {'Axial': {'PSNR': 13.63, 'SSIM': 0.618},
            'Coronal': {'PSNR': 15.44, 'SSIM': 0.699},
            'Sagittal': {'PSNR': 14.81, 'SSIM': 0.638}},
        8: {'Axial': {'PSNR': 16.89, 'SSIM': 0.639},
            'Coronal': {'PSNR': 18.12, 'SSIM': 0.724},
            'Sagittal': {'PSNR': 17.52, 'SSIM': 0.715}},
    },
    'FBPConvNet': {
        2: {'Axial': {'PSNR': 16.31, 'SSIM': 0.521},
            'Coronal': {'PSNR': 17.05, 'SSIM': 0.521},
            'Sagittal': {'PSNR': 16.07, 'SSIM': 0.483}},
        4: {'Axial': {'PSNR': 18.51, 'SSIM': 0.708},
            'Coronal': {'PSNR': 19.91, 'SSIM': 0.723},
            'Sagittal': {'PSNR': 18.36, 'SSIM': 0.711}},
        8: {'Axial': {'PSNR': 18.98, 'SSIM': 0.687},
            'Coronal': {'PSNR': 20.10, 'SSIM': 0.774},
            'Sagittal': {'PSNR': 18.71, 'SSIM': 0.741}},
    },
    'Lahiri et al.': {
        2: {'Axial': {'PSNR': 20.81, 'SSIM': 0.595},
            'Coronal': {'PSNR': 20.86, 'SSIM': 0.544},
            'Sagittal': {'PSNR': 19.68, 'SSIM': 0.498}},
        4: {'Axial': {'PSNR': 20.45, 'SSIM': 0.684},
            'Coronal': {'PSNR': 21.47, 'SSIM': 0.713},
            'Sagittal': {'PSNR': 20.54, 'SSIM': 0.689}},
        8: {'Axial': {'PSNR': 23.98, 'SSIM': 0.811},
            'Coronal': {'PSNR': 23.62, 'SSIM': 0.722},
            'Sagittal': {'PSNR': 22.11, 'SSIM': 0.727}},
    },
    'Chung et al.': {
        2: {'Axial': {'PSNR': 23.47, 'SSIM': 0.743},
            'Coronal': {'PSNR': 23.52, 'SSIM': 0.807},
            'Sagittal': {'PSNR': 22.48, 'SSIM': 0.709}},
        4: {'Axial': {'PSNR': 27.47, 'SSIM': 0.808},
            'Coronal': {'PSNR': 27.81, 'SSIM': 0.819},
            'Sagittal': {'PSNR': 26.30, 'SSIM': 0.753}},
        8: {'Axial': {'PSNR': 27.73, 'SSIM': 0.832},
            'Coronal': {'PSNR': 27.14, 'SSIM': 0.832},
            'Sagittal': {'PSNR': 26.48, 'SSIM': 0.726}},
    },
    'TGV_z': {
        2: {'Axial': {'PSNR': 24.62, 'SSIM': 0.631},
            'Coronal': {'PSNR': 25.46, 'SSIM': 0.611},
            'Sagittal': {'PSNR': 24.69, 'SSIM': 0.620}},
        4: {'Axial': {'PSNR': 26.15, 'SSIM': 0.774},
            'Coronal': {'PSNR': 27.42, 'SSIM': 0.783},
            'Sagittal': {'PSNR': 26.29, 'SSIM': 0.787}},
        8: {'Axial': {'PSNR': 25.16, 'SSIM': 0.705},
            'Coronal': {'PSNR': 25.72, 'SSIM': 0.715},
            'Sagittal': {'PSNR': 25.21, 'SSIM': 0.722}},
    },
    'TGV_xy': {
        2: {'Axial': {'PSNR': 23.52, 'SSIM': 0.711},
            'Coronal': {'PSNR': 24.83, 'SSIM': 0.710},
            'Sagittal': {'PSNR': 23.64, 'SSIM': 0.712}},
        4: {'Axial': {'PSNR': 21.56, 'SSIM': 0.699},
            'Coronal': {'PSNR': 22.92, 'SSIM': 0.705},
            'Sagittal': {'PSNR': 21.67, 'SSIM': 0.708}},
        8: {'Axial': {'PSNR': 25.72, 'SSIM': 0.797},
            'Coronal': {'PSNR': 26.30, 'SSIM': 0.809},
            'Sagittal': {'PSNR': 25.81, 'SSIM': 0.812}},
    },
    'TGV_xyz': {
        2: {'Axial': {'PSNR': 21.55, 'SSIM': 0.752},
            'Coronal': {'PSNR': 22.75, 'SSIM': 0.753},
            'Sagittal': {'PSNR': 21.65, 'SSIM': 0.754}},
        4: {'Axial': {'PSNR': 28.74, 'SSIM': 0.794},
            'Coronal': {'PSNR': 29.88, 'SSIM': 0.800},
            'Sagittal': {'PSNR': 29.03, 'SSIM': 0.804}},
        8: {'Axial': {'PSNR': 30.29, 'SSIM': 0.815},
            'Coronal': {'PSNR': 31.35, 'SSIM': 0.825},
            'Sagittal': {'PSNR': 30.57, 'SSIM': 0.829}},
    },
    'TV_z': {
        2: {'Axial': {'PSNR': 25.12, 'SSIM': 0.693},
            'Coronal': {'PSNR': 26.73, 'SSIM': 0.678},
            'Sagittal': {'PSNR': 25.34, 'SSIM': 0.684}},
        4: {'Axial': {'PSNR': 30.49, 'SSIM': 0.807},
            'Coronal': {'PSNR': 31.67, 'SSIM': 0.816},
            'Sagittal': {'PSNR': 30.72, 'SSIM': 0.820}},
        8: {'Axial': {'PSNR': 32.33, 'SSIM': 0.823},
            'Coronal': {'PSNR': 33.06, 'SSIM': 0.835},
            'Sagittal': {'PSNR': 32.47, 'SSIM': 0.840}},
    },
}



# 设置画布
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)


metrics = ['PSNR', 'SSIM']
view_labels = ['Axial', 'Sagittal', 'Coronal']

for col, metric in enumerate(metrics):
    for row, slice_type in enumerate(view_labels):
        ax = axes[row, col]
        for method_idx, method in enumerate(methods):
            # 获取每种方法在不同视角下的 PSNR 或 SSIM 值
            #values = [data1[method][view][slice_type][metric] for view in slices]
            #values = [data2[method][view][slice_type][metric] for view in slices]
            values = [data3[method][view][slice_type][metric] for view in slices]
            ax.plot(slices, values, marker='o', color=colors[method_idx], label=method, linewidth=2)

        ax.set_title(f'{metric} - {slice_type}')
        ax.set_xlabel('View')  # 横坐标为视角数
        ax.set_ylabel(metric)  # 纵坐标为指标
        # ax.set_xticks(slices)  # 设置横坐标刻度为 2, 4, 8
        # ax.set_xticklabels(slices)  # 显示刻度标签为 2, 4, 8
        ax.grid(True)
        ax.legend(loc='lower right', fontsize='small')  # 自动选择最佳位置
        # if row == 0 and col == 1:
        #     ax.legend(loc='upper right')

# 调整布局
plt.tight_layout()
plt.savefig("plot.png")  # 保存到当前目录