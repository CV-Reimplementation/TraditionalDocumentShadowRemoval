import numpy as np
from PIL import Image
from helper import *
import matplotlib.pyplot as plt
from skimage import color,filters,transform
from scipy import ndimage as ndi
from p_tqdm import p_map
import os

def main():

    folder = 'dataset/Kligler/test/input/'

    file_names = os.listdir(folder)

    file_names = [folder + f_name for f_name in file_names]

    p_map(process, file_names)

def process(filename):
    ip_img = load_img(filename)


    is_0_255 = True   #modifies algo if pixel values are 0-255 (False > color_mosaic), arg passed to functions.
    # f, ax = plot_img(ip_img, "Original Image")

    # Extracting global background colour

    choice = 1

    if choice==1:
        # Global average over each channel  (Approach #1)  
        I_global = get_global_colour_1(ip_img,is_0_255)
    elif choice==2:
        # Max pixel value for each channel  (Approach #2)
        I_global = get_global_colour_2(ip_img,is_0_255)
    elif choice==3:
        # Average of top 50 pixels          (Approach #3)
        I_global = get_global_colour_3(ip_img,is_0_255)
    # f,ax = show_img_compare(ip_img, I_global, 'Original Image', 'Estimated Global Background')

    # Estimation of Local Background
    p = 0.9
    block_size = 7
    I_local = get_local_bg(ip_img,p, block_size, is_0_255)
    # f,ax = plot_img(I_local, "Local Background")

    # Refined Estimation of Local Background
    threshold = 0.01
    median_block_size = 17
    I_local_refined = get_local_bg_refined(I_local, ip_img, threshold, median_block_size, is_0_255)
    # f,ax = plot_img(I_local_refined, "Refined Local Background")

    # Deshadowed Image
    generate_deshadow(ip_img, I_local, I_global, is_0_255)
    I_deshadow = generate_deshadow(ip_img, I_local, I_global, is_0_255)
    I_deshadow_refined = generate_deshadow(ip_img, I_local_refined, I_global, is_0_255)
    pil_image = Image.fromarray(np.uint8(I_deshadow_refined))
    pil_image.save(os.path.basename(filename))

if __name__ == '__main__':
    main()
