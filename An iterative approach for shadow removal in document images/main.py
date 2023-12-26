import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color,filters,transform
from scipy import ndimage as ndi
from helper import *
from p_tqdm import p_map
import os

def process(filename):
    ip_img = load_img(filename)


    is_0_255 = True   #modifies algo if pixel values are 0-255 (False > color_mosaic), arg passed to functions.

    ip_img_gray = color.rgb2gray(ip_img)
    threshold_mask = filters.threshold_local(ip_img_gray, block_size=2001)
    binary_img = ip_img_gray > threshold_mask

    window_size = 15
    n_iter = 2
    iter_img = ip_img
    iter_binary_img = binary_img
    for _ in range(n_iter):
        iter_img, iter_binary_img = estimate_shading_reflectance(iter_img, iter_binary_img, window_size)

    pil_image = Image.fromarray(np.uint8(iter_img))
    pil_image.save(os.path.join('result', os.path.basename(filename)))

def main():

    folder = './input/'

    os.makedirs('result', exist_ok=True)

    file_names = os.listdir(folder)

    file_names = [folder + f_name for f_name in file_names]

    p_map(process, file_names)


if __name__ == "__main__":
    main()