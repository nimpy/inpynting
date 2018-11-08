from data_structures import Image2BInpainted, Patch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from efficient_energy_optimization import initialization
from label_pruning import label_pruning
# import random
# import sys

image = None

patch_size = 0
gap = 0

THRESHOLD_UNCERTAINTY = 80000
MAX_NB_LABELS = 10

def loading_data():
    global image
    global patch_size
    global gap

    # inputs
    folder_path = '/home/niaki/Code/Inpainting_Tijana/images'
    image_filename = 'Lenna.png'
    mask_filename = 'Mask512.jpg'
    image_inpainted_name = 'Lenna'
    image_inpainted_version = '1st_try'
    patch_size = 16
    gap = 8

    # settings
    np.set_printoptions(threshold=np.nan)

    # loading the image and the mask
    image_rgb = imageio.imread(folder_path + '/' + image_filename)
    mask = imageio.imread(folder_path + '/' + mask_filename)
    mask = mask[:, :, 0]

    image = Image2BInpainted(image_rgb, mask)



def main():

    loading_data()

    plt.imshow(image.rgb, interpolation='nearest')
    plt.show()
    plt.imshow(image.mask, cmap='gray')
    plt.show()


    #mask = mask // 255



    nodes = initialization(image, patch_size, gap, THRESHOLD_UNCERTAINTY)


    label_pruning(nodes.nodes_count, nodes.nodes_priority, nodes.nodes_differences, nodes.nodes_labels, nodes.nodes_coords, image, patch_size, gap, THRESHOLD_UNCERTAINTY, MAX_NB_LABELS)





if __name__ == "__main__":
    main()
