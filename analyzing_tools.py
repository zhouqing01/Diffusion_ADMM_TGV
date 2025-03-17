import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import os
import re
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt

def normalize_image(image):
    """
    Normalize image data to the range 0-1.
    """
    if image.dtype == np.uint8:
        image = image / 255.0
    elif image.dtype == np.uint16:
        image = image / 65535.0
    elif image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() > 1.0:
            image = image / image.max()
    else:
        raise ValueError("Unsupported image data type: {}".format(image.dtype))

    return image

def load_recon_gt_to_volume(root_path, recon_file_dir='recon/', label_file_dir='label/', volume_file_dir='volume/', volume_file_name='volume.npy', gt_file_name='ground_truth.npy'):
    """load reconstructed volume from npy file"""
    def extract_number_from_filename(filename):  
        match = re.search(r'\d+', filename)  
        return int(match.group()) if match else float('inf')
    
    def sort_files_by_number_in_filename(directory):  
        files = os.listdir(directory)  

        files_with_numbers = [(extract_number_from_filename(filename), filename) for filename in files]  

        sorted_files = sorted(files_with_numbers, key=lambda x: x[0])  

        sorted_filenames = [filename for _, filename in sorted_files]  

        return sorted_filenames  

    imgs = []
    names = []

    name_list = sort_files_by_number_in_filename(root_path + recon_file_dir)
    for name in name_list:
        try:
            img = Image.open(root_path + recon_file_dir + name)
            names.append(name)
        except:
            continue

        img = img.convert('L')

        # convert shape to (z, c, x, y)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        imgs.append(img)

    imgs = normalize_image(np.array(imgs))

    labels = []
    for name in names:
        try:
            label = Image.open(root_path + label_file_dir + name)
        except:
            continue

        label = label.convert('L')

        # convert shape to (z, c, x, y)
        label = np.array(label)
        label = np.expand_dims(label, axis=0)
        labels.append(label)
    
    labels = normalize_image(np.array(labels))

    # save volume
    if not os.path.exists(root_path + volume_file_dir):
        os.makedirs(root_path + volume_file_dir)

    np.save(root_path + volume_file_dir + volume_file_name, imgs.squeeze())
    np.save(root_path + volume_file_dir + gt_file_name, labels.squeeze())

def load_volume(path, reconstruct_file_name, reference_file_name):
    """load reconstructed and reference volumes from npy files"""
    reconstructed_volume = normalize_image(np.load(path + reconstruct_file_name).squeeze())
    reference_volume = normalize_image(np.load(path + reference_file_name).squeeze())
    return reconstructed_volume, reference_volume

def calculate_metrics(slice_reconstructed, slice_reference):
    """calculate PSNR and SSIM for a single slice"""
    psnr_value = psnr(slice_reference, slice_reconstructed, data_range=1.0)
    ssim_value = ssim(slice_reference, slice_reconstructed, data_range=1.0)
    return psnr_value, ssim_value

def extract_slices_and_evaluate(volume_reconstructed, volume_reference):
    """extract slices from the reconstructed and reference volumes and calculate average PSNR and SSIM for each slice"""
    psnr_axial, ssim_axial = [], []
    psnr_coronal, ssim_coronal = [], []
    psnr_sagittal, ssim_sagittal = [], []

    # volume shape is (z, x, y)

    # axial slices
    for z in range(volume_reconstructed.shape[0]):
        slice_reconstructed = volume_reconstructed[z, :, :]
        slice_reference = volume_reference[z, :, :]
        p, s = calculate_metrics(slice_reconstructed, slice_reference)
        psnr_axial.append(p)
        ssim_axial.append(s)

    # coronal slices
    for x in range(volume_reconstructed.shape[1]):
        slice_reconstructed = volume_reconstructed[:, x, :]
        slice_reference = volume_reference[:, x, :]
        p, s = calculate_metrics(slice_reconstructed, slice_reference)
        psnr_coronal.append(p)
        ssim_coronal.append(s)

    # sagittal slices
    for y in range(volume_reconstructed.shape[2]):
        slice_reconstructed = volume_reconstructed[:, :, y]
        slice_reference = volume_reference[:, :, y]
        p, s = calculate_metrics(slice_reconstructed, slice_reference)
        psnr_sagittal.append(p)
        ssim_sagittal.append(s)

    return {
        "Axial": {"PSNR": np.mean(psnr_axial), "SSIM": np.mean(ssim_axial)},
        "Coronal": {"PSNR": np.mean(psnr_coronal), "SSIM": np.mean(ssim_coronal)},
        "Sagittal": {"PSNR": np.mean(psnr_sagittal), "SSIM": np.mean(ssim_sagittal)}
    }

def plot_comparison(reconstructed_volumes, titles, ground_truth, output_file=None, simple=True, inset=False):
    """generate a plot comparing the reconstructed and ground truth volumes along the three axes"""
    fig, axes = plt.subplots(3, len(reconstructed_volumes) + 1, figsize=((len(reconstructed_volumes) + 1) * 3, 9))

    str_psnr = '' if simple else 'PSNR:'
    str_ssim = '' if simple else 'SSIM:'

    for i, volume in enumerate(reconstructed_volumes):
        # get the axial, sagittal, and coronal slices
        axial_slice = volume[volume.shape[0] // 2, :, :]
        sagittal_slice = volume[:, volume.shape[1] // 2, :]
        coronal_slice = volume[:, :, volume.shape[2] // 2]

        # calculate PSNR and SSIM for each slice
        axial_gt = ground_truth[ground_truth.shape[0] // 2, :, :]
        sagittal_gt = ground_truth[:, ground_truth.shape[1] // 2, :]
        coronal_gt = ground_truth[:, :, ground_truth.shape[2] // 2]

        psnr_axial = psnr(axial_gt, axial_slice, data_range=1.0)
        ssim_axial = ssim(axial_gt, axial_slice, data_range=1.0)

        psnr_sagittal = psnr(sagittal_gt, sagittal_slice, data_range=1.0)
        ssim_sagittal = ssim(sagittal_gt, sagittal_slice, data_range=1.0)

        psnr_coronal = psnr(coronal_gt, coronal_slice, data_range=1.0)
        ssim_coronal = ssim(coronal_gt, coronal_slice, data_range=1.0)

        # transfrom to 256x256
        axial_slice = np.array(Image.fromarray(axial_slice).resize((256, 256)))
        sagittal_slice = np.array(Image.fromarray(sagittal_slice).resize((256, 256)))
        coronal_slice = np.array(Image.fromarray(coronal_slice).resize((256, 256)))

        axial_gt = np.array(Image.fromarray(axial_gt).resize((256, 256)))
        sagittal_gt = np.array(Image.fromarray(sagittal_gt).resize((256, 256)))
        coronal_gt = np.array(Image.fromarray(coronal_gt).resize((256, 256)))

        slices = [axial_slice, sagittal_slice, coronal_slice]
        gts = [axial_gt, sagittal_gt, coronal_gt]
        psnr_values = [psnr_axial, psnr_sagittal, psnr_coronal]
        ssim_values = [ssim_axial, ssim_sagittal, ssim_coronal]

        for j in range(3):
            axes[j, i].imshow(slices[j], cmap='gray')
            #axes[j, i].set_title(f'{titles[i]}\n{str_psnr} {psnr_values[j]:.2f} / {str_ssim} {ssim_values[j]:.3f}')
            axes[j, i].text(5, 30, f'{titles[i]}\n{str_psnr} {psnr_values[j]:.2f} / {str_ssim} {ssim_values[j]:.3f}', fontdict={'color': 'white'})
            axes[j, i].axis('off')

            if inset:
                # add inset with zoomed-in region
                inset_ax = axes[j, i].inset_axes([0.7, 0.0, 0.3, 0.3])  # [x, y, width, height]
                inset_ax.add_patch(plt.Rectangle((0, 0), 80, 80, edgecolor='yellow', facecolor='none', lw=2))
                inset_ax.imshow(slices[j][70:110, 70:110], cmap='gray')
                inset_ax.set_xlim(0, 40)
                inset_ax.set_ylim(40, 0)
                inset_ax.axis('off')

                # add rectangle around zoomed-in region
                axes[j, i].add_patch(plt.Rectangle((70, 70), 40, 40, edgecolor='yellow', facecolor='none', lw=1))

    # plot the ground truth
    for j in range(3):
        axes[j, -1].imshow(gts[j], cmap='gray')
        axes[j, -1].text(5, 30, 'Ground truth', fontdict={'color': 'white'})
        axes[j, -1].axis('off')

        if inset:
            # add inset with zoomed-in region
            inset_ax = axes[j, -1].inset_axes([0.7, 0.0, 0.3, 0.3])  # [x, y, width, height]
            inset_ax.add_patch(plt.Rectangle((0, 0), 80, 80, edgecolor='yellow', facecolor='none', lw=2))
            inset_ax.imshow(gts[j][70:110, 70:110], cmap='gray')
            inset_ax.set_xlim(0, 40)
            inset_ax.set_ylim(40, 0)
            inset_ax.axis('off')

            # add rectangle around zoomed-in region
            axes[j, -1].add_patch(plt.Rectangle((70, 70), 40, 40, edgecolor='yellow', facecolor='none', lw=1))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    if output_file:
        plt.savefig(output_file)

    plt.show()

def table_comparison(data, title, output_file=None):
    """
    create a table comparing the PSNR and SSIM values for the reconstructed volumes
    """
    df = pd.DataFrame(data)

    # set table style
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.axis('off')

    # create table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # set cell colors
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')

    plt.title(title)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)

    plt.show()
        
def flatten_nested_dict(nested_dict, parent_key='', sep=' '):
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def flatten_dicts_to_df(dict_list, name_list):
    assert len(dict_list) == len(name_list), 'Length of dict_list and name_list must be equal'

    list_format_dict = {'Method': name_list}
    for d in dict_list:
        for k, v in d.items():
            if k not in list_format_dict:
                list_format_dict[k] = []
            list_format_dict[k].append(v)
    return list_format_dict

